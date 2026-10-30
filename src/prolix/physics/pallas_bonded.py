"""Pallas kernels for 2-, 3-, 4-, and 5-body bonded & pair-correction potentials (R1-R4).

Provides:
  - ``pallas_bonds_2body``: harmonic bonds & Urey-Bradley (BLOCK_B = 64)
  - ``pallas_exceptions_14``: explicit AMBER 1-4 exception pairs (BLOCK_B = 64)
  - ``pallas_pme_exclusion_pairs``: PME reciprocal-space bonded exclusion correction (BLOCK_B = 64)
  - ``pallas_angles_harmonic``: 3-body harmonic angles (BLOCK_A = 64)
  - ``pallas_torsions_4body``: 4-body proper dihedrals, periodic & harmonic impropers (BLOCK_D = 64)
  - ``pallas_cmap_torsions``: 5-body CMAP 2D bicubic spline torsions (BLOCK_C = 32)
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import xtrax
from jax import custom_vjp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.scipy.special import erf
from jax_md import space
from xtrax.intent import Context
from xtrax.kernel_seam import VjpKernel

from prolix.export import in_export_context
from prolix.physics.cmap import compute_cmap_energies, precompute_cmap_coefficients
from prolix.physics.pbc import create_periodic_space

Vec3 = tuple[jax.Array, jax.Array, jax.Array]

_BLOCK_B = 64
_BLOCK_A = 64
_BLOCK_D = 64
_BLOCK_C = 32


def _triton_gpu_available() -> bool:
    """Return True iff default backend is GPU with compute capability >= 8.0 (Ampere+ for Triton)."""
    if jax.default_backend() != "gpu":
        return False
    try:
        dev = jax.devices("gpu")[0]
        cc = getattr(dev, "compute_capability", None)
        if cc is not None and float(cc) < 8.0:
            return False
    except Exception:
        return False
    return True


def _triton_erfc(x: jax.Array) -> jax.Array:
    """Float32-accurate erfc(x) for x >= 0 via Abramowitz & Stegun 7.1.26 (Triton-compatible)."""
    dtype = x.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    x_pos = jnp.maximum(x, zero)
    t = one / (one + jnp.asarray(0.3275911, dtype=dtype) * x_pos)
    poly = t * (
        jnp.asarray(0.254829592, dtype=dtype)
        + t
        * (
            jnp.asarray(-0.284496736, dtype=dtype)
            + t
            * (
                jnp.asarray(1.421413741, dtype=dtype)
                + t
                * (
                    jnp.asarray(-1.453152027, dtype=dtype)
                    + t * jnp.asarray(1.061405429, dtype=dtype)
                )
            )
        )
    )
    return poly * jnp.exp(zero - (x_pos * x_pos))


def _triton_erf(x: jax.Array) -> jax.Array:
    """Float32-accurate erf(x) for x >= 0 via 1 - _triton_erfc(x) (Triton-compatible)."""
    return jnp.asarray(1.0, dtype=x.dtype) - _triton_erfc(x)


def _sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(s: jax.Array, a: Vec3) -> Vec3:
    return (s * a[0], s * a[1], s * a[2])


def _dot3(a: Vec3, b: Vec3) -> jax.Array:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _min_image_1d(d: jax.Array, box_len: jax.Array, has_box: bool) -> jax.Array:
    """Zero-box-safe 1D minimum-image PBC displacement (Invariant 5)."""
    if not has_box:
        return d
    dtype = d.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    b_safe = jnp.where(box_len > zero, box_len, one)
    return jnp.where(box_len > zero, d - box_len * jnp.floor(d / b_safe + half), d)


def _min_image3(
    d: Vec3,
    box_vec: tuple[jax.Array, ...] | jax.Array,
    has_box: bool,
) -> Vec3:
    if not has_box:
        return d
    n_b = len(box_vec) if isinstance(box_vec, tuple) else box_vec.shape[-1]
    if n_b == 3:
        return (
            _min_image_1d(d[0], box_vec[0], has_box),
            _min_image_1d(d[1], box_vec[1], has_box),
            _min_image_1d(d[2], box_vec[2], has_box),
        )
    dtype = d[0].dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    ax, by, cz, bx, cx, cy = (
        box_vec[0],
        box_vec[1],
        box_vec[2],
        box_vec[3],
        box_vec[4],
        box_vec[5],
    )
    dx, dy, dz = d[0], d[1], d[2]

    cz_pos = cz > zero
    cz_s = jnp.where(cz_pos, cz, one)
    sz = jnp.where(cz_pos, jnp.floor(dz / cz_s + half), zero)
    dz = dz - sz * cz
    dy = dy - sz * cy
    dx = dx - sz * cx

    by_pos = by > zero
    by_s = jnp.where(by_pos, by, one)
    sy = jnp.where(by_pos, jnp.floor(dy / by_s + half), zero)
    dy = dy - sy * by
    dx = dx - sy * bx

    ax_pos = ax > zero
    ax_s = jnp.where(ax_pos, ax, one)
    sx = jnp.where(ax_pos, jnp.floor(dx / ax_s + half), zero)
    dx = dx - sx * ax
    return (dx, dy, dz)


def _load_box6(box_ref) -> tuple[jax.Array, ...]:
    """Load all (3 or 6) box components from a 1D Pallas box_ref."""
    return tuple(
        _pl_load(box_ref, (jnp.full((1,), k, dtype=jnp.int32),))[0]
        for k in range(box_ref.shape[0])
    )


def _pl_atomic_add_1d(
    ref,
    idx_1d: jax.Array,
    val_1d: jax.Array,
    mask_1d: jax.Array,
    interpret: bool,
) -> None:
    """1D atomic add that serializes duplicate-index updates under interpret=True (Invariant 3)."""
    zero = jnp.asarray(0.0, dtype=val_1d.dtype)
    val_masked = jnp.where(mask_1d, val_1d, zero)
    if interpret:
        block_size = idx_1d.shape[0]

        def _serial_body(k, _):
            plt.atomic_add(ref, (idx_1d[k],), val_masked[k])

        jax.lax.fori_loop(0, block_size, _serial_body, None)
    else:
        plt.atomic_add(ref, (idx_1d,), val_masked, mask=mask_1d)


def _normalize_box(box: jax.Array | None, positions: jax.Array) -> tuple[jax.Array, bool]:
    """Normalize optional box (None, (3,), (6,), or lower-triangular (3, 3)) to (6,) or (B, 6) array [ax, by, cz, bx, cx, cy]."""
    dtype = positions.dtype
    if box is None:
        b = jnp.zeros((6,), dtype=dtype)
        has_box = False
    else:
        b_raw = jnp.asarray(box, dtype=dtype)
        if b_raw.ndim >= 2 and b_raw.shape[-2:] == (3, 3):
            ax = b_raw[..., 0, 0]
            by = b_raw[..., 1, 1]
            cz = b_raw[..., 2, 2]
            bx = b_raw[..., 1, 0]
            cx = b_raw[..., 2, 0]
            cy = b_raw[..., 2, 1]
            b = jnp.stack([ax, by, cz, bx, cx, cy], axis=-1)
        elif b_raw.shape[-1] == 3:
            b = jnp.concatenate([b_raw, jnp.zeros_like(b_raw)], axis=-1)
        else:
            b = b_raw
        has_box = True
    if b.ndim == 1 and positions.ndim == 3:
        b = jnp.broadcast_to(b[None, :], (positions.shape[0], b.shape[-1]))
    return b, has_box


def _make_disp_fn(box: jax.Array | None) -> space.DisplacementFn:
    """Create a JAX-MD displacement_fn for export-context fallbacks."""
    if box is None:
        return space.free()[0]
    disp_fn, _ = create_periodic_space(box)
    return disp_fn


def _min_image_host(dr: jax.Array, box: jax.Array, has_box: bool) -> jax.Array:
    """Host-side zero-box-safe minimum image displacement for parameter gradients."""
    if not has_box:
        return dr
    dx, dy, dz = _min_image3((dr[..., 0], dr[..., 1], dr[..., 2]), box, has_box)
    return jnp.stack([dx, dy, dz], axis=-1)


def _unwrap_primal(x):
    """Unwrap CustomVJPPrimal(value, perturbed) when called under symbolic_zeros=True."""
    return x.value if hasattr(x, "perturbed") else x


def _is_perturbed(x, default: bool = True) -> bool:
    """Return True if x is an actively differentiated CustomVJPPrimal (or default when raw)."""
    return bool(x.perturbed) if hasattr(x, "perturbed") else bool(default)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class _StaticShapesMeta:
    """Zero-leaf PyTree node holding static (shape, dtype) metadata across jit/vjp boundaries."""

    specs: tuple[tuple[tuple[int, ...], Any], ...]

    def __getitem__(self, idx):
        res = self.specs[idx]
        if isinstance(idx, slice):
            return tuple(jax.ShapeDtypeStruct(s, d) for s, d in res)
        return jax.ShapeDtypeStruct(res[0], res[1])

    def __iter__(self):
        for s, d in self.specs:
            yield jax.ShapeDtypeStruct(s, d)

    def __len__(self) -> int:
        return len(self.specs)


def _make_shapes_meta(*tensors) -> _StaticShapesMeta:
    return _StaticShapesMeta(tuple((t.shape, t.dtype) for t in tensors))


# =============================================================================
# R1.1: 2-Body Harmonic Bonds & Urey-Bradley (pallas_bonds_2body)
# =============================================================================


def _pallas_bonds_2body_single(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    dist_eps: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    B_real = indices.shape[0]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N

    n_blocks = max(1, (B_real + _BLOCK_B - 1) // _BLOCK_B)
    B_pad = n_blocks * _BLOCK_B
    pad_b = B_pad - B_real

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    idx_i = indices[:, 0].astype(jnp.int32)
    idx_j = indices[:, 1].astype(jnp.int32)
    r0 = params[:, 0].astype(dtype)
    k_val = params[:, 1].astype(dtype)
    mask_f = mask.astype(dtype)

    if pad_b > 0:
        idx_i = jnp.pad(idx_i, (0, pad_b), constant_values=-1)
        idx_j = jnp.pad(idx_j, (0, pad_b), constant_values=-1)
        r0 = jnp.pad(r0, (0, pad_b))
        k_val = jnp.pad(k_val, (0, pad_b))
        mask_f = jnp.pad(mask_f, (0, pad_b))

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    box_vec = jnp.asarray(box.astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _bonds_kernel(
        px_ref,
        py_ref,
        pz_ref,
        idx_i_ref,
        idx_j_ref,
        r0_ref,
        k_ref,
        mask_ref,
        box_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        offs = pid * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)

        a_i = _pl_load(idx_i_ref, (offs,))
        a_j = _pl_load(idx_j_ref, (offs,))
        m_val = _pl_load(mask_ref, (offs,))
        valid = (offs < B_real) & (m_val > zero) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N)
        m_eff = jnp.where(valid, m_val, zero)

        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)

        r_i = (_pl_load(px_ref, (safe_i,)), _pl_load(py_ref, (safe_i,)), _pl_load(pz_ref, (safe_i,)))
        r_j = (_pl_load(px_ref, (safe_j,)), _pl_load(py_ref, (safe_j,)), _pl_load(pz_ref, (safe_j,)))

        d_vec = _min_image3(_sub3(r_i, r_j), _load_box6(box_ref), has_box)
        d_sq = _dot3(d_vec, d_vec)
        if dist_eps > 0.0:
            dist = jnp.sqrt(d_sq + jnp.asarray(dist_eps, dtype=dtype))
        else:
            d_sq_safe = jnp.where(d_sq > zero, d_sq, one)
            dist = jnp.where(d_sq > zero, jnp.sqrt(d_sq_safe), zero)

        r0_b = _pl_load(r0_ref, (offs,))
        k_b = _pl_load(k_ref, (offs,))
        diff = dist - r0_b
        e_b = half * m_eff * k_b * diff * diff
        _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_b)[None])

        dist_denom = jnp.where(dist > zero, dist, one)
        f_coeff = jnp.where(valid & (dist > zero), (zero - m_eff * k_b * diff) / dist_denom, zero)
        f_i = _scale3(f_coeff, d_vec)
        f_j = _scale3(zero - one, f_i)

        _pl_atomic_add_1d(fx_out_ref, safe_i, f_i[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_i, f_i[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_i, f_i[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_j, f_j[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_j, f_j[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_j, f_j[2], valid, interpret)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(12)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((n_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _bonds_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        input_output_aliases={9: 1, 10: 2, 11: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_bonds_2body",
    )(px, py, pz, idx_i, idx_j, r0, k_val, mask_f, box_vec, zero_f, zero_f, zero_f)
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _bonds_2body_param_grads(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    dist_eps: float,
) -> jax.Array:
    N = positions.shape[0]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    a_i = indices[:, 0].astype(jnp.int32)
    a_j = indices[:, 1].astype(jnp.int32)
    m_f = mask.astype(dtype)
    valid = (m_f > zero) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N)
    m_eff = jnp.where(valid, m_f, zero)
    max_atom = max(N - 1, 0)
    safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
    safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
    dr = _min_image_host(positions[safe_i] - positions[safe_j], box, has_box)
    d_sq = jnp.sum(dr * dr, axis=-1)
    if dist_eps > 0.0:
        dist = jnp.sqrt(d_sq + jnp.asarray(dist_eps, dtype=dtype))
    else:
        d_sq_safe = jnp.where(d_sq > zero, d_sq, one)
        dist = jnp.where(d_sq > zero, jnp.sqrt(d_sq_safe), zero)
    diff = dist - params[:, 0].astype(dtype)
    k_b = params[:, 1].astype(dtype)
    grad_r0 = -m_eff * k_b * diff
    grad_k = half * m_eff * diff * diff
    return jnp.stack([grad_r0, grad_k], axis=-1).astype(params.dtype)


def _pallas_bonds_2body_vmappable(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    dist_eps: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    hb = bool(has_box)
    de = float(dist_eps)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, idx, prm, msk, bx):
        return _pallas_bonds_2body_single(pos, idx, prm, msk, bx, hb, de, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, idx, prm, msk, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                idx[b_idx] if in_batched[1] else idx,
                prm[b_idx] if in_batched[2] else prm,
                msk[b_idx] if in_batched[3] else msk,
                bx[b_idx] if in_batched[4] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                indices[b] if indices.ndim == 3 else indices,
                params[b] if params.ndim == 3 else params,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _vmappable(positions, indices, params, mask, box)


@functools.partial(custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _pallas_bonds_2body_diff(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    dist_eps: float,
    interpret: bool,
    compute_param_grads: bool | None = None,
) -> jax.Array:
    energy, _ = _pallas_bonds_2body_vmappable(
        positions, indices, params, mask, box, has_box, dist_eps, interpret
    )
    return energy


def _pallas_bonds_2body_fwd(
    positions,
    indices,
    params,
    mask,
    box,
    has_box,
    dist_eps,
    interpret,
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    idx_v = _unwrap_primal(indices)
    prm_v = _unwrap_primal(params)
    msk_v = _unwrap_primal(mask)
    bx_v = _unwrap_primal(box)

    energy, forces = _pallas_bonds_2body_vmappable(
        pos_v, idx_v, prm_v, msk_v, bx_v, has_box, dist_eps, interpret
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    any_param_perturbed = (
        bool(compute_param_grads)
        if compute_param_grads is not None
        else _is_perturbed(params, default_pert)
    )
    shapes_meta = _make_shapes_meta(pos_v, idx_v, prm_v, msk_v, bx_v)
    param_res = (pos_v, idx_v, prm_v, msk_v, bx_v) if any_param_perturbed else None
    return energy, (forces, shapes_meta, param_res)


def _pallas_bonds_2body_bwd(has_box, dist_eps, interpret, compute_param_grads, res, g):
    del interpret, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    positions, indices, params, mask, box = param_res
    if positions.ndim == 2:
        gp = _bonds_2body_param_grads(positions, indices, params, mask, box, has_box, dist_eps)
        grad_params = (g * gp).astype(params.dtype)
    else:
        B = positions.shape[0]
        b_idx = indices.ndim > 2
        b_prm = params.ndim > 2
        b_msk = mask.ndim > 1
        b_box = box.ndim > 1

        def _step(b):
            gp_b = _bonds_2body_param_grads(
                positions[b],
                indices[b] if b_idx else indices,
                params[b] if b_prm else params,
                mask[b] if b_msk else mask,
                box[b] if b_box else box,
                has_box,
                dist_eps,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (g_b * gp_b).astype(params.dtype)

        gp_all = jax.lax.map(_step, jnp.arange(B))
        grad_params = gp_all if b_prm else jnp.sum(gp_all, axis=0)

    return (
        grad_pos,
        jnp.zeros(shapes_meta[1].shape, dtype=shapes_meta[1].dtype),
        grad_params,
        jnp.zeros(shapes_meta[3].shape, dtype=shapes_meta[3].dtype),
        jnp.zeros(shapes_meta[4].shape, dtype=shapes_meta[4].dtype),
    )


_pallas_bonds_2body_diff.defvjp(
    _pallas_bonds_2body_fwd, _pallas_bonds_2body_bwd, symbolic_zeros=True
)


def bonds_2body_reference(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    *,
    has_box: bool = False,
    dist_eps: float = 1e-10,
) -> jax.Array:
    """Plain-JAX 2-body harmonic bond & Urey-Bradley reference matching ``pallas_bonds_2body``."""

    def _one(pos_1: jax.Array, idx_1: jax.Array, prm_1: jax.Array, msk_1: jax.Array, bx_1: jax.Array) -> jax.Array:
        N = pos_1.shape[0]
        dtype = pos_1.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        a_i = idx_1[:, 0].astype(jnp.int32)
        a_j = idx_1[:, 1].astype(jnp.int32)
        m_f = msk_1.astype(dtype)
        valid = (m_f > zero) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N)
        m_eff = jnp.where(valid, m_f, zero)
        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
        dr = _min_image_host(pos_1[safe_i] - pos_1[safe_j], bx_1, has_box)
        d_sq = jnp.sum(dr * dr, axis=-1)
        if dist_eps > 0.0:
            dist = jnp.sqrt(d_sq + jnp.asarray(dist_eps, dtype=dtype))
        else:
            d_sq_safe = jnp.where(d_sq > zero, d_sq, one)
            dist = jnp.where(d_sq > zero, jnp.sqrt(d_sq_safe), zero)
        diff = dist - prm_1[:, 0].astype(dtype)
        return jnp.sum(jnp.asarray(0.5, dtype=dtype) * m_eff * prm_1[:, 1].astype(dtype) * diff * diff)

    if positions.ndim == 3:
        B = positions.shape[0]
        b_idx = indices.ndim == 3
        b_prm = params.ndim == 3
        b_msk = mask.ndim == 2
        b_box = box.ndim == 2
        return jax.vmap(
            lambda b: _one(
                positions[b],
                indices[b] if b_idx else indices,
                params[b] if b_prm else params,
                mask[b] if b_msk else mask,
                box[b] if b_box else box,
            )
        )(jnp.arange(B))
    return _one(positions, indices, params, mask, box)


def make_bonds_2body_op(
    *,
    has_box: bool = False,
    dist_eps: float = 1e-10,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for 2-body harmonic bonds & Urey-Bradley.

    Signature of the returned op: ``(positions, indices, params, mask, box) -> energy``.
    """
    hb = bool(has_box)
    de = float(dist_eps)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(bonds_2body_reference, has_box=hb, dist_eps=de)

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(pos, idx, prm, msk, bx):
            return _pallas_bonds_2body_fwd(pos, idx, prm, msk, bx, hb, de, interp, cpg)

        def _bwd(res, ct):
            return _pallas_bonds_2body_bwd(hb, de, interp, cpg, res, ct)

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_bonds_2body",
        context=context,
    )


def pallas_bonds_2body(
    positions: jax.Array,
    indices: jax.Array | None,
    params: jax.Array | None,
    mask: jax.Array | None = None,
    box: jax.Array | None = None,
    *,
    dist_eps: float = 1e-10,
    interpret: bool | None = None,
    compute_param_grads: bool | None = None,
    return_forces: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """1D-tiled (BLOCK_B=64) harmonic bond & Urey-Bradley Pallas kernel."""
    if indices is None or params is None or indices.shape[-2] == 0:
        z_e = jnp.zeros(positions.shape[:-2], dtype=positions.dtype)
        return (z_e, jnp.zeros_like(positions)) if return_forces else z_e

    idx_2col = jnp.asarray(indices[..., :, :2], dtype=jnp.int32)
    prm_arr = jnp.asarray(params, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(idx_2col.shape[:-1], dtype=positions.dtype)
        if mask is None
        else jnp.asarray(mask, dtype=positions.dtype)
    )
    box_arr, has_box = _normalize_box(box, positions)

    if return_forces:
        if interpret is None:
            interpret = not _triton_gpu_available()
        return _pallas_bonds_2body_vmappable(
            positions,
            idx_2col,
            prm_arr,
            msk_arr,
            box_arr,
            bool(has_box),
            float(dist_eps),
            bool(interpret),
        )

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_bonds_2body_op(
            has_box=bool(has_box),
            dist_eps=float(dist_eps),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(positions, idx_2col, prm_arr, msk_arr, box_arr)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_bonds_2body_diff(
        positions, idx_2col, prm_arr, msk_arr, box_arr, bool(has_box), float(dist_eps), bool(interpret), cpg
    )


# =============================================================================
# R1.2: Explicit AMBER 1-4 Exception Pairs (pallas_exceptions_14)
# =============================================================================


def _pallas_exceptions_14_single(
    positions: jax.Array,
    pairs: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    chargeprods: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    lam: jax.Array,
    has_box: bool,
    coulomb_constant: float,
    softcore_alpha: float,
    softcore_exponent: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    E_real = pairs.shape[0]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N

    n_blocks = max(1, (E_real + _BLOCK_B - 1) // _BLOCK_B)
    E_pad = n_blocks * _BLOCK_B
    pad_e = E_pad - E_real

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    idx_i = pairs[:, 0].astype(jnp.int32)
    idx_j = pairs[:, 1].astype(jnp.int32)
    sig_f = sigmas.astype(dtype)
    eps_f = epsilons.astype(dtype)
    qp_f = chargeprods.astype(dtype)
    mask_f = mask.astype(dtype)

    if pad_e > 0:
        idx_i = jnp.pad(idx_i, (0, pad_e), constant_values=-1)
        idx_j = jnp.pad(idx_j, (0, pad_e), constant_values=-1)
        sig_f = jnp.pad(sig_f, (0, pad_e), constant_values=1.0)
        eps_f = jnp.pad(eps_f, (0, pad_e))
        qp_f = jnp.pad(qp_f, (0, pad_e))
        mask_f = jnp.pad(mask_f, (0, pad_e))

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    box_vec = jnp.asarray(box.astype(dtype))
    lam_vec = jnp.asarray(lam.reshape((1,)).astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _exc14_kernel(
        px_ref,
        py_ref,
        pz_ref,
        idx_i_ref,
        idx_j_ref,
        sig_ref,
        eps_ref,
        qp_ref,
        mask_ref,
        box_ref,
        lam_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        offs = pid * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)
        six = jnp.asarray(6.0, dtype=dtype)
        coul = jnp.asarray(coulomb_constant, dtype=dtype)

        a_i = _pl_load(idx_i_ref, (offs,))
        a_j = _pl_load(idx_j_ref, (offs,))
        m_val = _pl_load(mask_ref, (offs,))
        valid = (offs < E_real) & (m_val > zero) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N) & (a_i != a_j)
        m_eff = jnp.where(valid, m_val, zero)

        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)

        r_i = (_pl_load(px_ref, (safe_i,)), _pl_load(py_ref, (safe_i,)), _pl_load(pz_ref, (safe_i,)))
        r_j = (_pl_load(px_ref, (safe_j,)), _pl_load(py_ref, (safe_j,)), _pl_load(pz_ref, (safe_j,)))

        lam_v = _pl_load(lam_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]

        d_vec = _min_image3(_sub3(r_i, r_j), _load_box6(box_ref), has_box)
        r_sq = _dot3(d_vec, d_vec) + jnp.asarray(1e-12, dtype=dtype)
        dist = jnp.sqrt(r_sq)

        sig_e = _pl_load(sig_ref, (offs,))
        eps_e = _pl_load(eps_ref, (offs,))
        qp_e = _pl_load(qp_ref, (offs,))

        sig_safe = jnp.where(valid, jnp.maximum(sig_e, jnp.asarray(1e-4, dtype=dtype)), one)
        one_minus_lam = jnp.maximum(one - lam_v, zero)
        soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * (one_minus_lam ** int(softcore_exponent))
        r6 = (dist / sig_safe) ** 6
        denom = soft_term + r6 + jnp.asarray(1e-12, dtype=dtype)
        inv_d = one / denom
        inv_d2 = inv_d * inv_d
        inv_d3 = inv_d2 * inv_d
        e_lj = four * eps_e * lam_v * (inv_d2 - inv_d)
        e_coul = coul * qp_e / dist
        e_pair = m_eff * (e_lj + e_coul)
        _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_pair)[None])

        # dE/dr / r = m_eff / r^2 * (4 * eps * lam * (1/denom^2 - 2/denom^3) * 6 * r6 - C * qp / r)
        de_dr_over_r = (m_eff / r_sq) * (
            four * eps_e * lam_v * (inv_d2 - two * inv_d3) * (six * r6) - e_coul
        )
        f_coeff = jnp.where(valid, zero - de_dr_over_r, zero)
        f_i = _scale3(f_coeff, d_vec)
        f_j = _scale3(zero - one, f_i)

        _pl_atomic_add_1d(fx_out_ref, safe_i, f_i[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_i, f_i[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_i, f_i[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_j, f_j[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_j, f_j[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_j, f_j[2], valid, interpret)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(14)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((n_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _exc14_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        input_output_aliases={11: 1, 12: 2, 13: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_exceptions_14",
    )(px, py, pz, idx_i, idx_j, sig_f, eps_f, qp_f, mask_f, box_vec, lam_vec, zero_f, zero_f, zero_f)
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _exceptions_14_param_grads(
    positions: jax.Array,
    pairs: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    chargeprods: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    lam: jax.Array,
    has_box: bool,
    coulomb_constant: float,
    softcore_alpha: float,
    softcore_exponent: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    N = positions.shape[0]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    a_i = pairs[:, 0].astype(jnp.int32)
    a_j = pairs[:, 1].astype(jnp.int32)
    m_f = mask.astype(dtype)
    valid = (m_f > zero) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N) & (a_i != a_j)
    m_eff = jnp.where(valid, m_f, zero)
    max_atom = max(N - 1, 0)
    safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
    safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
    dr = _min_image_host(positions[safe_i] - positions[safe_j], box, has_box)
    dist = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=dtype))
    sig_e = sigmas.astype(dtype)
    eps_e = epsilons.astype(dtype)
    sig_safe = jnp.where(valid, jnp.maximum(sig_e, jnp.asarray(1e-4, dtype=dtype)), one)
    lam_v = lam.reshape(()).astype(dtype)
    one_minus_lam = jnp.maximum(one - lam_v, zero)
    p_exp = int(softcore_exponent)
    soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * (one_minus_lam ** p_exp)
    r6 = (dist / sig_safe) ** 6
    denom = soft_term + r6 + jnp.asarray(1e-12, dtype=dtype)
    inv_d = one / denom
    inv_d2 = inv_d * inv_d
    inv_d3 = inv_d2 * inv_d
    de_ddenom = 4.0 * eps_e * lam_v * (inv_d2 - 2.0 * inv_d3)
    grad_sig = m_eff * de_ddenom * (-6.0 * r6 / sig_safe)
    grad_eps = m_eff * 4.0 * lam_v * (inv_d2 - inv_d)
    grad_qp = m_eff * jnp.asarray(coulomb_constant, dtype=dtype) / dist
    dsoft_dlam = jnp.where(
        lam_v < one,
        -jnp.asarray(softcore_alpha * p_exp, dtype=dtype) * (one_minus_lam ** max(p_exp - 1, 0)),
        zero,
    )
    grad_lam = jnp.sum(m_eff * (4.0 * eps_e * (inv_d2 - inv_d) + de_ddenom * dsoft_dlam)).reshape(lam.shape)
    return (
        grad_sig.astype(sigmas.dtype),
        grad_eps.astype(epsilons.dtype),
        grad_qp.astype(chargeprods.dtype),
        grad_lam.astype(lam.dtype),
    )


def _pallas_exceptions_14_vmappable(
    positions,
    pairs,
    sigmas,
    epsilons,
    chargeprods,
    mask,
    box,
    lam,
    has_box,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    interpret,
):
    hb = bool(has_box)
    cc = float(coulomb_constant)
    sa = float(softcore_alpha)
    se = int(softcore_exponent)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, pr, sig, eps, qp, msk, bx, lm):
        return _pallas_exceptions_14_single(pos, pr, sig, eps, qp, msk, bx, lm, hb, cc, sa, se, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, pr, sig, eps, qp, msk, bx, lm):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                pr[b_idx] if in_batched[1] else pr,
                sig[b_idx] if in_batched[2] else sig,
                eps[b_idx] if in_batched[3] else eps,
                qp[b_idx] if in_batched[4] else qp,
                msk[b_idx] if in_batched[5] else msk,
                bx[b_idx] if in_batched[6] else bx,
                lm[b_idx] if in_batched[7] else lm,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                pairs[b] if pairs.ndim == 3 else pairs,
                sigmas[b] if sigmas.ndim == 2 else sigmas,
                epsilons[b] if epsilons.ndim == 2 else epsilons,
                chargeprods[b] if chargeprods.ndim == 2 else chargeprods,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
                lam[b] if lam.ndim >= 1 else lam,
            )
        )(jnp.arange(B))
    return _vmappable(positions, pairs, sigmas, epsilons, chargeprods, mask, box, lam)


@functools.partial(custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13))
def _pallas_exceptions_14_diff(
    positions,
    pairs,
    sigmas,
    epsilons,
    chargeprods,
    mask,
    box,
    lam,
    has_box,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    interpret,
    compute_param_grads=None,
):
    energy, _ = _pallas_exceptions_14_vmappable(
        positions,
        pairs,
        sigmas,
        epsilons,
        chargeprods,
        mask,
        box,
        lam,
        has_box,
        coulomb_constant,
        softcore_alpha,
        softcore_exponent,
        interpret,
    )
    return energy


def _pallas_exceptions_14_fwd(
    positions,
    pairs,
    sigmas,
    epsilons,
    chargeprods,
    mask,
    box,
    lam,
    has_box,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    interpret,
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    pr_v = _unwrap_primal(pairs)
    sig_v = _unwrap_primal(sigmas)
    eps_v = _unwrap_primal(epsilons)
    qp_v = _unwrap_primal(chargeprods)
    msk_v = _unwrap_primal(mask)
    bx_v = _unwrap_primal(box)
    lam_v = _unwrap_primal(lam)

    energy, forces = _pallas_exceptions_14_vmappable(
        pos_v,
        pr_v,
        sig_v,
        eps_v,
        qp_v,
        msk_v,
        bx_v,
        lam_v,
        has_box,
        coulomb_constant,
        softcore_alpha,
        softcore_exponent,
        interpret,
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    if compute_param_grads is not None:
        any_param_perturbed = bool(compute_param_grads)
    else:
        any_param_perturbed = (
            _is_perturbed(sigmas, default_pert)
            or _is_perturbed(epsilons, default_pert)
            or _is_perturbed(chargeprods, default_pert)
            or _is_perturbed(lam, default_pert)
        )
    shapes_meta = _make_shapes_meta(
        pos_v, pr_v, sig_v, eps_v, qp_v, msk_v, bx_v, lam_v
    )
    param_res = (
        (pos_v, pr_v, sig_v, eps_v, qp_v, msk_v, bx_v, lam_v)
        if any_param_perturbed
        else None
    )
    return energy, (forces, shapes_meta, param_res)


def _pallas_exceptions_14_bwd(
    has_box,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    interpret,
    compute_param_grads,
    res,
    g,
):
    del interpret, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    positions, pairs, sigmas, epsilons, chargeprods, mask, box, lam = param_res
    if positions.ndim == 2:
        gs, ge, gq, glam = _exceptions_14_param_grads(
            positions,
            pairs,
            sigmas,
            epsilons,
            chargeprods,
            mask,
            box,
            lam,
            has_box,
            coulomb_constant,
            softcore_alpha,
            softcore_exponent,
        )
        grad_sig = (g * gs).astype(sigmas.dtype)
        grad_eps = (g * ge).astype(epsilons.dtype)
        grad_qp = (g * gq).astype(chargeprods.dtype)
        grad_lam = (g * glam).astype(lam.dtype)
    else:
        B = positions.shape[0]
        b_pr = pairs.ndim > 2
        b_sig = sigmas.ndim > 1
        b_eps = epsilons.ndim > 1
        b_qp = chargeprods.ndim > 1
        b_msk = mask.ndim > 1
        b_box = box.ndim > 1
        b_lam = lam.ndim > 0

        def _step(b):
            gs_b, ge_b, gq_b, glam_b = _exceptions_14_param_grads(
                positions[b],
                pairs[b] if b_pr else pairs,
                sigmas[b] if b_sig else sigmas,
                epsilons[b] if b_eps else epsilons,
                chargeprods[b] if b_qp else chargeprods,
                mask[b] if b_msk else mask,
                box[b] if b_box else box,
                lam[b] if b_lam else lam,
                has_box,
                coulomb_constant,
                softcore_alpha,
                softcore_exponent,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (
                (g_b * gs_b).astype(sigmas.dtype),
                (g_b * ge_b).astype(epsilons.dtype),
                (g_b * gq_b).astype(chargeprods.dtype),
                (g_b * glam_b).astype(lam.dtype),
            )

        gs_all, ge_all, gq_all, glam_all = jax.lax.map(_step, jnp.arange(B))
        grad_sig = gs_all if b_sig else jnp.sum(gs_all, axis=0)
        grad_eps = ge_all if b_eps else jnp.sum(ge_all, axis=0)
        grad_qp = gq_all if b_qp else jnp.sum(gq_all, axis=0)
        grad_lam = glam_all if b_lam else jnp.sum(glam_all, axis=0)

    return (
        grad_pos,
        jnp.zeros(shapes_meta[1].shape, dtype=shapes_meta[1].dtype),
        grad_sig,
        grad_eps,
        grad_qp,
        jnp.zeros(shapes_meta[5].shape, dtype=shapes_meta[5].dtype),
        jnp.zeros(shapes_meta[6].shape, dtype=shapes_meta[6].dtype),
        grad_lam,
    )


_pallas_exceptions_14_diff.defvjp(
    _pallas_exceptions_14_fwd, _pallas_exceptions_14_bwd, symbolic_zeros=True
)


def exceptions_14_reference(
    positions: jax.Array,
    pairs: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    chargeprods: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    lam: jax.Array,
    *,
    has_box: bool = False,
    coulomb_constant: float = 332.0637,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
) -> jax.Array:
    """Plain-JAX AMBER 1-4 exception pair reference matching ``pallas_exceptions_14``."""

    def _one(pos_1, pr_1, s_1, e_1, qp_1, m_1, bx_1, l_1):
        N = pos_1.shape[0]
        dtype = pos_1.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        a_i = pr_1[:, 0].astype(jnp.int32)
        a_j = pr_1[:, 1].astype(jnp.int32)
        m_f = m_1.astype(dtype)
        valid = (m_f > zero) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N) & (a_i != a_j)
        m_eff = jnp.where(valid, m_f, zero)
        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
        dr = _min_image_host(pos_1[safe_i] - pos_1[safe_j], bx_1, has_box)
        dist = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=dtype))
        sig_safe = jnp.where(valid, jnp.maximum(s_1.astype(dtype), jnp.asarray(1e-4, dtype=dtype)), one)
        lam_v = l_1.reshape(()).astype(dtype)
        one_minus_lam = jnp.maximum(one - lam_v, zero)
        soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * (one_minus_lam ** int(softcore_exponent))
        r6 = (dist / sig_safe) ** 6
        denom = soft_term + r6 + jnp.asarray(1e-12, dtype=dtype)
        e_lj = jnp.asarray(4.0, dtype=dtype) * e_1.astype(dtype) * lam_v * (one / (denom * denom) - one / denom)
        e_coul = jnp.asarray(coulomb_constant, dtype=dtype) * qp_1.astype(dtype) / dist
        return jnp.sum((e_lj + e_coul) * m_eff)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _one(
                positions[b],
                pairs[b] if pairs.ndim == 3 else pairs,
                sigmas[b] if sigmas.ndim == 2 else sigmas,
                epsilons[b] if epsilons.ndim == 2 else epsilons,
                chargeprods[b] if chargeprods.ndim == 2 else chargeprods,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
                lam[b] if lam.ndim >= 1 else lam,
            )
        )(jnp.arange(B))
    return _one(positions, pairs, sigmas, epsilons, chargeprods, mask, box, lam)


def make_exceptions_14_op(
    *,
    has_box: bool = False,
    coulomb_constant: float = 332.0637,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for AMBER 1-4 exception pairs.

    Signature of the returned op:
    ``(positions, pairs, sigmas, epsilons, chargeprods, mask, box, lam) -> energy``.
    """
    hb = bool(has_box)
    cc = float(coulomb_constant)
    sa = float(softcore_alpha)
    se = int(softcore_exponent)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(
        exceptions_14_reference,
        has_box=hb,
        coulomb_constant=cc,
        softcore_alpha=sa,
        softcore_exponent=se,
    )

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(pos, pr, sig, eps, qp, msk, bx, lm):
            return _pallas_exceptions_14_fwd(
                pos, pr, sig, eps, qp, msk, bx, lm, hb, cc, sa, se, interp, cpg
            )

        def _bwd(res, ct):
            return _pallas_exceptions_14_bwd(hb, cc, sa, se, interp, cpg, res, ct)

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_exceptions_14",
        context=context,
    )


def pallas_exceptions_14(
    positions: jax.Array,
    exception_pairs: jax.Array | None,
    exception_sigmas: jax.Array | None,
    exception_epsilons: jax.Array | None,
    exception_chargeprods: jax.Array | None,
    mask: jax.Array | None = None,
    box: jax.Array | None = None,
    coulomb_constant: float = 332.0637,
    *,
    soft_core_lambda: float | jax.Array | None = None,
    softcore_lambda: float | jax.Array | None = None,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    interpret: bool | None = None,
    compute_param_grads: bool | None = None,
) -> jax.Array:
    """1D-tiled (BLOCK_B=64) AMBER 1-4 exception pair Pallas kernel."""
    if (
        exception_pairs is None
        or exception_sigmas is None
        or exception_epsilons is None
        or exception_chargeprods is None
        or exception_pairs.shape[-2] == 0
    ):
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    eff_lam = soft_core_lambda if soft_core_lambda is not None else (softcore_lambda if softcore_lambda is not None else 1.0)
    lam_arr = jnp.asarray(eff_lam, dtype=positions.dtype)
    pairs_arr = jnp.asarray(exception_pairs[..., :, :2], dtype=jnp.int32)
    sig_arr = jnp.asarray(exception_sigmas, dtype=positions.dtype)
    eps_arr = jnp.asarray(exception_epsilons, dtype=positions.dtype)
    qp_arr = jnp.asarray(exception_chargeprods, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(pairs_arr.shape[:-1], dtype=positions.dtype)
        if mask is None
        else jnp.asarray(mask, dtype=positions.dtype)
    )
    box_arr, has_box = _normalize_box(box, positions)

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_exceptions_14_op(
            has_box=bool(has_box),
            coulomb_constant=float(coulomb_constant),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(positions, pairs_arr, sig_arr, eps_arr, qp_arr, msk_arr, box_arr, lam_arr)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_exceptions_14_diff(
        positions,
        pairs_arr,
        sig_arr,
        eps_arr,
        qp_arr,
        msk_arr,
        box_arr,
        lam_arr,
        bool(has_box),
        float(coulomb_constant),
        float(softcore_alpha),
        int(softcore_exponent),
        bool(interpret),
        cpg,
    )


# =============================================================================
# R1.3: PME Bonded Exclusion Pair Correction (pallas_pme_exclusion_pairs)
# =============================================================================


def _pallas_pme_exclusion_pairs_single(
    positions: jax.Array,
    excl_pair_indices: jax.Array,
    excl_pair_scales_elec: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    coulomb_constant: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    P_real = excl_pair_indices.shape[0]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N

    n_blocks = max(1, (P_real + _BLOCK_B - 1) // _BLOCK_B)
    P_pad = n_blocks * _BLOCK_B
    pad_p = P_pad - P_real

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    q_pad = jnp.pad(charges.astype(dtype), (0, pad_n)) if pad_n > 0 else charges.astype(dtype)
    mask_pad = (
        jnp.pad(atom_mask.astype(jnp.int32), (0, pad_n), constant_values=0)
        if pad_n > 0
        else atom_mask.astype(jnp.int32)
    )

    idx_i = excl_pair_indices[:, 0].astype(jnp.int32)
    idx_j = excl_pair_indices[:, 1].astype(jnp.int32)
    s_elec = excl_pair_scales_elec.astype(dtype)
    if pad_p > 0:
        idx_i = jnp.pad(idx_i, (0, pad_p), constant_values=-1)
        idx_j = jnp.pad(idx_j, (0, pad_p), constant_values=-1)
        s_elec = jnp.pad(s_elec, (0, pad_p), constant_values=1.0)

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    box_vec = jnp.asarray(box.astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _pme_excl_kernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        atom_mask_ref,
        idx_i_ref,
        idx_j_ref,
        s_elec_ref,
        box_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        offs = pid * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        coul = jnp.asarray(coulomb_constant, dtype=dtype)
        alpha = jnp.asarray(pme_alpha, dtype=dtype)
        two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

        a_i = _pl_load(idx_i_ref, (offs,))
        a_j = _pl_load(idx_j_ref, (offs,))
        in_range = (offs < P_real) & (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N) & (a_i != a_j)
        max_atom = max(N - 1, 0)
        safe_i = jnp.where(in_range, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(in_range, jnp.clip(a_j, 0, max_atom), 0)

        m_i = _pl_load(atom_mask_ref, (safe_i,)) > 0
        m_j = _pl_load(atom_mask_ref, (safe_j,)) > 0
        valid = in_range & m_i & m_j

        r_i = (_pl_load(px_ref, (safe_i,)), _pl_load(py_ref, (safe_i,)), _pl_load(pz_ref, (safe_i,)))
        r_j = (_pl_load(px_ref, (safe_j,)), _pl_load(py_ref, (safe_j,)), _pl_load(pz_ref, (safe_j,)))

        d_vec = _min_image3(_sub3(r_i, r_j), _load_box6(box_ref), has_box)
        r_sq_raw = _dot3(d_vec, d_vec)
        r_raw = jnp.where(r_sq_raw > zero, jnp.sqrt(jnp.where(r_sq_raw > zero, r_sq_raw, one)), zero)
        r_p = r_raw + jnp.asarray(1e-6, dtype=dtype)

        q_i = _pl_load(q_ref, (safe_i,))
        q_j = _pl_load(q_ref, (safe_j,))
        s_p = _pl_load(s_elec_ref, (offs,))

        alpha_r = alpha * r_p
        erf_val = erf(alpha_r) if (interpret and dtype == jnp.float64) else _triton_erf(alpha_r)
        pref = (zero - jnp.where(valid, one - s_p, zero)) * coul * q_i * q_j
        e_p = pref * erf_val / r_p
        _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_p)[None])

        # dE_p / dr_p = pref * (2*alpha/sqrt(pi) * r_p * exp(-alpha^2 r_p^2) - erf(alpha r_p)) / r_p^2
        de_dr_full = pref * (alpha * two_over_sqrt_pi * r_p * jnp.exp(zero - alpha_r * alpha_r) - erf_val) / (r_p * r_p)
        de_dr_taylor = (
            pref
            * (alpha ** 3)
            * two_over_sqrt_pi
            * (zero - r_p)
            * (jnp.asarray(2.0 / 3.0, dtype=dtype) - jnp.asarray(2.0 / 5.0, dtype=dtype) * (alpha_r * alpha_r))
        )
        de_dr = jnp.where(alpha_r < jnp.asarray(0.05, dtype=dtype), de_dr_taylor, de_dr_full)
        r_denom = jnp.maximum(r_raw, jnp.asarray(1e-12, dtype=dtype))
        f_coeff = jnp.where(valid & (r_raw > zero), (zero - de_dr) / r_denom, zero)
        f_i = _scale3(f_coeff, d_vec)
        f_j = _scale3(zero - one, f_i)

        _pl_atomic_add_1d(fx_out_ref, safe_i, f_i[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_i, f_i[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_i, f_i[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_j, f_j[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_j, f_j[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_j, f_j[2], valid, interpret)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(12)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((n_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _pme_excl_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        input_output_aliases={9: 1, 10: 2, 11: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_pme_exclusion_pairs",
    )(px, py, pz, q_pad, mask_pad, idx_i, idx_j, s_elec, box_vec, zero_f, zero_f, zero_f)
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _pme_excl_pairs_param_grads(
    positions: jax.Array,
    excl_pair_indices: jax.Array,
    excl_pair_scales_elec: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    coulomb_constant: float,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    a_i = excl_pair_indices[:, 0].astype(jnp.int32)
    a_j = excl_pair_indices[:, 1].astype(jnp.int32)
    in_range = (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N) & (a_i != a_j)
    max_atom = max(N - 1, 0)
    safe_i = jnp.where(in_range, jnp.clip(a_i, 0, max_atom), 0)
    safe_j = jnp.where(in_range, jnp.clip(a_j, 0, max_atom), 0)
    mask_bool = jnp.asarray(atom_mask, dtype=jnp.bool_)
    valid = in_range & mask_bool[safe_i] & mask_bool[safe_j]

    dr = _min_image_host(positions[safe_i] - positions[safe_j], box, has_box)
    d_sq = jnp.sum(dr * dr, axis=-1)
    r_raw = jnp.where(d_sq > zero, jnp.sqrt(jnp.where(d_sq > zero, d_sq, one)), zero)
    r_p = r_raw + jnp.asarray(1e-6, dtype=dtype)
    erf_over_r = erf(jnp.asarray(pme_alpha, dtype=dtype) * r_p) / r_p

    q_f = charges.astype(dtype)
    s_f = excl_pair_scales_elec.astype(dtype)
    coul = jnp.asarray(coulomb_constant, dtype=dtype)
    w_valid = jnp.where(valid, one, zero)

    grad_scale = w_valid * coul * q_f[safe_i] * q_f[safe_j] * erf_over_r
    pair_factor = -w_valid * (one - s_f) * coul * erf_over_r
    dq_i = pair_factor * q_f[safe_j]
    dq_j = pair_factor * q_f[safe_i]
    grad_q = jnp.zeros((N,), dtype=dtype).at[safe_i].add(dq_i).at[safe_j].add(dq_j)
    return grad_scale.astype(excl_pair_scales_elec.dtype), grad_q.astype(charges.dtype)


def _pallas_pme_exclusion_pairs_vmappable(
    positions,
    excl_pair_indices,
    excl_pair_scales_elec,
    charges,
    atom_mask,
    box,
    has_box,
    pme_alpha,
    coulomb_constant,
    interpret,
):
    hb = bool(has_box)
    pa = float(pme_alpha)
    cc = float(coulomb_constant)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, pairs, scales, q, msk, bx):
        return _pallas_pme_exclusion_pairs_single(pos, pairs, scales, q, msk, bx, hb, pa, cc, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, pairs, scales, q, msk, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                pairs[b_idx] if in_batched[1] else pairs,
                scales[b_idx] if in_batched[2] else scales,
                q[b_idx] if in_batched[3] else q,
                msk[b_idx] if in_batched[4] else msk,
                bx[b_idx] if in_batched[5] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                excl_pair_indices[b] if excl_pair_indices.ndim == 3 else excl_pair_indices,
                excl_pair_scales_elec[b] if excl_pair_scales_elec.ndim == 2 else excl_pair_scales_elec,
                charges[b] if charges.ndim == 2 else charges,
                atom_mask[b] if atom_mask.ndim == 2 else atom_mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _vmappable(positions, excl_pair_indices, excl_pair_scales_elec, charges, atom_mask, box)


@functools.partial(custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _pallas_pme_exclusion_pairs_diff(
    positions,
    excl_pair_indices,
    excl_pair_scales_elec,
    charges,
    atom_mask,
    box,
    has_box,
    pme_alpha,
    coulomb_constant,
    interpret,
    compute_param_grads=None,
):
    energy, _ = _pallas_pme_exclusion_pairs_vmappable(
        positions,
        excl_pair_indices,
        excl_pair_scales_elec,
        charges,
        atom_mask,
        box,
        has_box,
        pme_alpha,
        coulomb_constant,
        interpret,
    )
    return energy


def _pallas_pme_exclusion_pairs_fwd(
    positions,
    excl_pair_indices,
    excl_pair_scales_elec,
    charges,
    atom_mask,
    box,
    has_box,
    pme_alpha,
    coulomb_constant,
    interpret,
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    pr_v = _unwrap_primal(excl_pair_indices)
    sc_v = _unwrap_primal(excl_pair_scales_elec)
    q_v = _unwrap_primal(charges)
    msk_v = _unwrap_primal(atom_mask)
    bx_v = _unwrap_primal(box)

    energy, forces = _pallas_pme_exclusion_pairs_vmappable(
        pos_v,
        pr_v,
        sc_v,
        q_v,
        msk_v,
        bx_v,
        has_box,
        pme_alpha,
        coulomb_constant,
        interpret,
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    if compute_param_grads is not None:
        any_param_perturbed = bool(compute_param_grads)
    else:
        any_param_perturbed = _is_perturbed(
            excl_pair_scales_elec, default_pert
        ) or _is_perturbed(charges, default_pert)

    shapes_meta = _make_shapes_meta(pos_v, pr_v, sc_v, q_v, msk_v, bx_v)
    param_res = (
        (pos_v, pr_v, sc_v, q_v, msk_v, bx_v) if any_param_perturbed else None
    )
    return energy, (forces, shapes_meta, param_res)


def _pallas_pme_exclusion_pairs_bwd(
    has_box, pme_alpha, coulomb_constant, interpret, compute_param_grads, res, g
):
    del interpret, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    positions, excl_pair_indices, excl_pair_scales_elec, charges, atom_mask, box = param_res
    if positions.ndim == 2:
        gs, gq = _pme_excl_pairs_param_grads(
            positions,
            excl_pair_indices,
            excl_pair_scales_elec,
            charges,
            atom_mask,
            box,
            has_box,
            pme_alpha,
            coulomb_constant,
        )
        grad_scales = (g * gs).astype(excl_pair_scales_elec.dtype)
        grad_q = (g * gq).astype(charges.dtype)
    else:
        B = positions.shape[0]
        b_pr = excl_pair_indices.ndim > 2
        b_sc = excl_pair_scales_elec.ndim > 1
        b_q = charges.ndim > 1
        b_msk = atom_mask.ndim > 1
        b_box = box.ndim > 1

        def _step(b):
            gs_b, gq_b = _pme_excl_pairs_param_grads(
                positions[b],
                excl_pair_indices[b] if b_pr else excl_pair_indices,
                excl_pair_scales_elec[b] if b_sc else excl_pair_scales_elec,
                charges[b] if b_q else charges,
                atom_mask[b] if b_msk else atom_mask,
                box[b] if b_box else box,
                has_box,
                pme_alpha,
                coulomb_constant,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (
                (g_b * gs_b).astype(excl_pair_scales_elec.dtype),
                (g_b * gq_b).astype(charges.dtype),
            )

        gs_all, gq_all = jax.lax.map(_step, jnp.arange(B))
        grad_scales = gs_all if b_sc else jnp.sum(gs_all, axis=0)
        grad_q = gq_all if b_q else jnp.sum(gq_all, axis=0)

    return (
        grad_pos,
        jnp.zeros(shapes_meta[1].shape, dtype=shapes_meta[1].dtype),
        grad_scales,
        grad_q,
        jnp.zeros(shapes_meta[4].shape, dtype=shapes_meta[4].dtype),
        jnp.zeros(shapes_meta[5].shape, dtype=shapes_meta[5].dtype),
    )


_pallas_pme_exclusion_pairs_diff.defvjp(
    _pallas_pme_exclusion_pairs_fwd,
    _pallas_pme_exclusion_pairs_bwd,
    symbolic_zeros=True,
)


def pme_exclusion_pairs_reference(
    positions: jax.Array,
    excl_pair_indices: jax.Array,
    excl_pair_scales_elec: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box: jax.Array,
    *,
    has_box: bool = False,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
) -> jax.Array:
    """Plain-JAX PME bonded exclusion pair reference matching ``pallas_pme_exclusion_pairs``."""

    def _one(pos_1, pr_1, sc_1, q_1, m_1, bx_1):
        N = pos_1.shape[0]
        dtype = pos_1.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        a_i = pr_1[:, 0].astype(jnp.int32)
        a_j = pr_1[:, 1].astype(jnp.int32)
        in_r = (a_i >= 0) & (a_i < N) & (a_j >= 0) & (a_j < N) & (a_i != a_j)
        max_atom = max(N - 1, 0)
        safe_i = jnp.where(in_r, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(in_r, jnp.clip(a_j, 0, max_atom), 0)
        m_bool = jnp.asarray(m_1, dtype=jnp.bool_)
        valid = in_r & m_bool[safe_i] & m_bool[safe_j]
        dr = _min_image_host(pos_1[safe_i] - pos_1[safe_j], bx_1, has_box)
        d_sq = jnp.sum(dr * dr, axis=-1)
        r_raw = jnp.where(d_sq > zero, jnp.sqrt(jnp.where(d_sq > zero, d_sq, one)), zero)
        dist = r_raw + jnp.asarray(1e-6, dtype=dtype)
        e_pair = (
            -(one - sc_1.astype(dtype))
            * jnp.asarray(coulomb_constant, dtype=dtype)
            * q_1.astype(dtype)[safe_i]
            * q_1.astype(dtype)[safe_j]
            * erf(jnp.asarray(pme_alpha, dtype=dtype) * dist)
            / dist
        )
        return jnp.sum(jnp.where(valid, e_pair, zero))

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _one(
                positions[b],
                excl_pair_indices[b] if excl_pair_indices.ndim == 3 else excl_pair_indices,
                excl_pair_scales_elec[b] if excl_pair_scales_elec.ndim == 2 else excl_pair_scales_elec,
                charges[b] if charges.ndim == 2 else charges,
                atom_mask[b] if atom_mask.ndim == 2 else atom_mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _one(positions, excl_pair_indices, excl_pair_scales_elec, charges, atom_mask, box)


def make_pme_exclusion_pairs_op(
    *,
    has_box: bool = False,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for PME reciprocal-space bonded exclusion pairs.

    Signature of the returned op:
    ``(positions, excl_pair_indices, excl_pair_scales_elec, charges, atom_mask, box) -> energy``.
    """
    hb = bool(has_box)
    pa = float(pme_alpha)
    cc = float(coulomb_constant)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(
        pme_exclusion_pairs_reference,
        has_box=hb,
        pme_alpha=pa,
        coulomb_constant=cc,
    )

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(pos, pairs, scales, q, msk, bx):
            return _pallas_pme_exclusion_pairs_fwd(
                pos, pairs, scales, q, msk, bx, hb, pa, cc, interp, cpg
            )

        def _bwd(res, ct):
            return _pallas_pme_exclusion_pairs_bwd(hb, pa, cc, interp, cpg, res, ct)

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_pme_exclusion_pairs",
        context=context,
    )


def pallas_pme_exclusion_pairs(
    positions: jax.Array,
    excl_pair_indices: jax.Array | None,
    excl_pair_scales_elec: jax.Array | None,
    charges: jax.Array,
    atom_mask: jax.Array | None = None,
    box: jax.Array | None = None,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    *,
    n_atoms: int | None = None,
    is_row_format: bool = False,
    interpret: bool | None = None,
    compute_param_grads: bool | None = None,
) -> jax.Array:
    """1D-tiled (BLOCK_B=64) PME reciprocal-space bonded exclusion correction Pallas kernel."""
    if (
        excl_pair_indices is None
        or excl_pair_scales_elec is None
        or excl_pair_indices.shape[-2] == 0
        or excl_pair_indices.shape[-1] == 0
    ):
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    if is_row_format:
        N_row = excl_pair_indices.shape[-2]
        M_row = excl_pair_indices.shape[-1]
        i_grid = jnp.broadcast_to(
            jnp.arange(N_row, dtype=jnp.int32)[:, None],
            excl_pair_indices.shape,
        )
        pairs_arr = jnp.stack([i_grid, jnp.asarray(excl_pair_indices, dtype=jnp.int32)], axis=-1).reshape(
            *excl_pair_indices.shape[:-2], N_row * M_row, 2
        )
        scales_arr = jnp.asarray(excl_pair_scales_elec, dtype=positions.dtype).reshape(
            *excl_pair_scales_elec.shape[:-2], N_row * M_row
        )
        eff_coul = float(coulomb_constant) * 0.5
    else:
        pairs_arr = jnp.asarray(excl_pair_indices[..., :, :2], dtype=jnp.int32)
        scales_arr = jnp.asarray(excl_pair_scales_elec, dtype=positions.dtype)
        eff_coul = float(coulomb_constant)

    q_arr = jnp.asarray(charges, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(positions.shape[:-1], dtype=jnp.bool_)
        if atom_mask is None
        else jnp.asarray(atom_mask, dtype=jnp.bool_)
    )
    if n_atoms is not None:
        bound_mask = jnp.arange(positions.shape[-2]) < int(n_atoms)
        msk_arr = msk_arr & bound_mask
    box_arr, has_box = _normalize_box(box, positions)

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_pme_exclusion_pairs_op(
            has_box=bool(has_box),
            pme_alpha=float(pme_alpha),
            coulomb_constant=float(eff_coul),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(positions, pairs_arr, scales_arr, q_arr, msk_arr, box_arr)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_pme_exclusion_pairs_diff(
        positions,
        pairs_arr,
        scales_arr,
        q_arr,
        msk_arr,
        box_arr,
        bool(has_box),
        float(pme_alpha),
        float(eff_coul),
        bool(interpret),
        cpg,
    )


# =============================================================================
# R2: 3-Body Harmonic Angle Kernel (pallas_angles_harmonic)
# =============================================================================


def _pallas_angles_harmonic_single(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    dist_eps: float,
    interpret: bool,
    angle_regularization_mode: str = "legacy_denom_eps",
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    A_real = indices.shape[0]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N

    n_blocks = max(1, (A_real + _BLOCK_A - 1) // _BLOCK_A)
    A_pad = n_blocks * _BLOCK_A
    pad_a = A_pad - A_real

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    idx_i = indices[:, 0].astype(jnp.int32)
    idx_j = indices[:, 1].astype(jnp.int32)
    idx_k = indices[:, 2].astype(jnp.int32)
    theta0 = params[:, 0].astype(dtype)
    k_val = params[:, 1].astype(dtype)
    mask_f = mask.astype(dtype)

    if pad_a > 0:
        idx_i = jnp.pad(idx_i, (0, pad_a), constant_values=-1)
        idx_j = jnp.pad(idx_j, (0, pad_a), constant_values=-1)
        idx_k = jnp.pad(idx_k, (0, pad_a), constant_values=-1)
        theta0 = jnp.pad(theta0, (0, pad_a))
        k_val = jnp.pad(k_val, (0, pad_a))
        mask_f = jnp.pad(mask_f, (0, pad_a))

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    box_vec = jnp.asarray(box.astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _angles_kernel(
        px_ref,
        py_ref,
        pz_ref,
        idx_i_ref,
        idx_j_ref,
        idx_k_ref,
        theta0_ref,
        k_ref,
        mask_ref,
        box_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        offs = pid * _BLOCK_A + jnp.arange(_BLOCK_A, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)

        a_i = _pl_load(idx_i_ref, (offs,))
        a_j = _pl_load(idx_j_ref, (offs,))
        a_k = _pl_load(idx_k_ref, (offs,))
        m_val = _pl_load(mask_ref, (offs,))
        valid = (
            (offs < A_real)
            & (m_val > zero)
            & (a_i >= 0)
            & (a_i < N)
            & (a_j >= 0)
            & (a_j < N)
            & (a_k >= 0)
            & (a_k < N)
        )
        m_eff = jnp.where(valid, m_val, zero)

        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
        safe_k = jnp.where(valid, jnp.clip(a_k, 0, max_atom), 0)

        r_i = (_pl_load(px_ref, (safe_i,)), _pl_load(py_ref, (safe_i,)), _pl_load(pz_ref, (safe_i,)))
        r_j = (_pl_load(px_ref, (safe_j,)), _pl_load(py_ref, (safe_j,)), _pl_load(pz_ref, (safe_j,)))
        r_k = (_pl_load(px_ref, (safe_k,)), _pl_load(py_ref, (safe_k,)), _pl_load(pz_ref, (safe_k,)))

        box_xyz = _load_box6(box_ref)

        v_ji = _min_image3(_sub3(r_i, r_j), box_xyz, has_box)
        v_jk = _min_image3(_sub3(r_k, r_j), box_xyz, has_box)

        d_sq_ji = _dot3(v_ji, v_ji)
        d_sq_jk = _dot3(v_jk, v_jk)
        if dist_eps > 0.0:
            eps_val = jnp.asarray(dist_eps, dtype=dtype)
            d_ji_sq_eff = d_sq_ji + eps_val
            d_jk_sq_eff = d_sq_jk + eps_val
            d_ji = jnp.sqrt(d_ji_sq_eff)
            d_jk = jnp.sqrt(d_jk_sq_eff)
        else:
            d_ji_sq_eff = jnp.where(d_sq_ji > zero, d_sq_ji, one)
            d_jk_sq_eff = jnp.where(d_sq_jk > zero, d_sq_jk, one)
            d_ji = jnp.where(d_sq_ji > zero, jnp.sqrt(d_ji_sq_eff), zero)
            d_jk = jnp.where(d_sq_jk > zero, jnp.sqrt(d_jk_sq_eff), zero)

        dot_ik = _dot3(v_ji, v_jk)
        cp = _cross3(v_ji, v_jk)
        th0 = _pl_load(theta0_ref, (offs,))
        k_a = _pl_load(k_ref, (offs,))

        if angle_regularization_mode == "openmm_cross":
            # O3.2 OpenMM angleForce.cc cross-product angle & force projection
            cp_sq = _dot3(cp, cp)
            r_cp = jnp.sqrt(jnp.maximum(cp_sq, jnp.asarray(1e-24, dtype=dtype)))
            theta = jnp.arctan2(r_cp, dot_ik)
            dtheta = theta - th0
            e_a = half * m_eff * k_a * dtheta * dtheta
            _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_a)[None])

            neg_de_dth = jnp.where(
                valid & (cp_sq > jnp.asarray(1e-24, dtype=dtype)) & (d_ji > zero) & (d_jk > zero),
                zero - m_eff * k_a * dtheta,
                zero,
            )
            f_i = _scale3(neg_de_dth / (d_ji_sq_eff * r_cp), _cross3(v_ji, cp))
            f_k = _scale3(neg_de_dth / (d_jk_sq_eff * r_cp), _cross3(cp, v_jk))
            f_j = _scale3(zero - one, _add3(f_i, f_k))
        else:
            # O3.2 legacy_denom_eps: cross-product direction with regularized denominator
            denom = d_ji * d_jk + jnp.asarray(1e-8, dtype=dtype)
            c_raw = dot_ik / denom
            c_lim = jnp.asarray(0.999999, dtype=dtype)
            c = jnp.clip(c_raw, zero - c_lim, c_lim)
            theta = jnp.arccos(c)
            dtheta = theta - th0
            e_a = half * m_eff * k_a * dtheta * dtheta
            _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_a)[None])

            in_clip = (c_raw > (zero - c_lim)) & (c_raw < c_lim)
            sin_th = jnp.sqrt(jnp.maximum(one - c * c, jnp.asarray(1e-12, dtype=dtype)))
            neg_de_dth = jnp.where(
                valid & in_clip & (d_ji > zero) & (d_jk > zero),
                zero - m_eff * k_a * dtheta,
                zero,
            )
            r_cp_reg = denom * sin_th
            f_i = _scale3(neg_de_dth / (d_ji_sq_eff * r_cp_reg), _cross3(v_ji, cp))
            f_k = _scale3(neg_de_dth / (d_jk_sq_eff * r_cp_reg), _cross3(cp, v_jk))
            f_j = _scale3(zero - one, _add3(f_i, f_k))

        _pl_atomic_add_1d(fx_out_ref, safe_i, f_i[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_i, f_i[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_i, f_i[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_j, f_j[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_j, f_j[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_j, f_j[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_k, f_k[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_k, f_k[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_k, f_k[2], valid, interpret)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(13)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((n_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _angles_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        input_output_aliases={10: 1, 11: 2, 12: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_angles_harmonic",
    )(px, py, pz, idx_i, idx_j, idx_k, theta0, k_val, mask_f, box_vec, zero_f, zero_f, zero_f)
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _angles_harmonic_param_grads(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    dist_eps: float,
) -> jax.Array:
    N = positions.shape[0]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    a_i = indices[:, 0].astype(jnp.int32)
    a_j = indices[:, 1].astype(jnp.int32)
    a_k = indices[:, 2].astype(jnp.int32)
    m_f = mask.astype(dtype)
    valid = (
        (m_f > zero)
        & (a_i >= 0)
        & (a_i < N)
        & (a_j >= 0)
        & (a_j < N)
        & (a_k >= 0)
        & (a_k < N)
    )
    m_eff = jnp.where(valid, m_f, zero)
    max_atom = max(N - 1, 0)
    safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
    safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
    safe_k = jnp.where(valid, jnp.clip(a_k, 0, max_atom), 0)

    v_ji = _min_image_host(positions[safe_i] - positions[safe_j], box, has_box)
    v_jk = _min_image_host(positions[safe_k] - positions[safe_j], box, has_box)
    d_sq_ji = jnp.sum(v_ji * v_ji, axis=-1)
    d_sq_jk = jnp.sum(v_jk * v_jk, axis=-1)
    if dist_eps > 0.0:
        d_ji = jnp.sqrt(d_sq_ji + jnp.asarray(dist_eps, dtype=dtype))
        d_jk = jnp.sqrt(d_sq_jk + jnp.asarray(dist_eps, dtype=dtype))
    else:
        d_ji = jnp.where(d_sq_ji > zero, jnp.sqrt(jnp.where(d_sq_ji > zero, d_sq_ji, one)), zero)
        d_jk = jnp.where(d_sq_jk > zero, jnp.sqrt(jnp.where(d_sq_jk > zero, d_sq_jk, one)), zero)

    denom = d_ji * d_jk + jnp.asarray(1e-8, dtype=dtype)
    c_lim = jnp.asarray(0.999999, dtype=dtype)
    cos_th = jnp.clip(jnp.sum(v_ji * v_jk, axis=-1) / denom, -c_lim, c_lim)
    theta = jnp.arccos(cos_th)
    dtheta = theta - params[:, 0].astype(dtype)
    k_a = params[:, 1].astype(dtype)
    grad_th0 = -m_eff * k_a * dtheta
    grad_k = half * m_eff * dtheta * dtheta
    return jnp.stack([grad_th0, grad_k], axis=-1).astype(params.dtype)


def _pallas_angles_harmonic_vmappable(
    positions,
    indices,
    params,
    mask,
    box,
    has_box,
    dist_eps,
    interpret,
    angle_regularization_mode="legacy_denom_eps",
):
    hb = bool(has_box)
    de = float(dist_eps)
    interp = bool(interpret)
    arm = str(angle_regularization_mode)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, idx, prm, msk, bx):
        return _pallas_angles_harmonic_single(pos, idx, prm, msk, bx, hb, de, interp, arm)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, idx, prm, msk, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                idx[b_idx] if in_batched[1] else idx,
                prm[b_idx] if in_batched[2] else prm,
                msk[b_idx] if in_batched[3] else msk,
                bx[b_idx] if in_batched[4] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                indices[b] if indices.ndim == 3 else indices,
                params[b] if params.ndim == 3 else params,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _vmappable(positions, indices, params, mask, box)


@functools.partial(custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _pallas_angles_harmonic_diff(
    positions,
    indices,
    params,
    mask,
    box,
    has_box,
    dist_eps,
    interpret,
    angle_regularization_mode="legacy_denom_eps",
    compute_param_grads=None,
):
    energy, _ = _pallas_angles_harmonic_vmappable(
        positions,
        indices,
        params,
        mask,
        box,
        has_box,
        dist_eps,
        interpret,
        angle_regularization_mode,
    )
    return energy


def _pallas_angles_harmonic_fwd(
    positions,
    indices,
    params,
    mask,
    box,
    has_box,
    dist_eps,
    interpret,
    angle_regularization_mode="legacy_denom_eps",
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    idx_v = _unwrap_primal(indices)
    prm_v = _unwrap_primal(params)
    msk_v = _unwrap_primal(mask)
    bx_v = _unwrap_primal(box)

    energy, forces = _pallas_angles_harmonic_vmappable(
        pos_v,
        idx_v,
        prm_v,
        msk_v,
        bx_v,
        has_box,
        dist_eps,
        interpret,
        angle_regularization_mode,
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    any_param_perturbed = (
        bool(compute_param_grads)
        if compute_param_grads is not None
        else _is_perturbed(params, default_pert)
    )
    shapes_meta = _make_shapes_meta(pos_v, idx_v, prm_v, msk_v, bx_v)
    param_res = (pos_v, idx_v, prm_v, msk_v, bx_v) if any_param_perturbed else None
    return energy, (forces, shapes_meta, param_res)


def _pallas_angles_harmonic_bwd(
    has_box,
    dist_eps,
    interpret,
    angle_regularization_mode,
    compute_param_grads,
    res,
    g,
):
    del interpret, angle_regularization_mode, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    positions, indices, params, mask, box = param_res
    if positions.ndim == 2:
        gp = _angles_harmonic_param_grads(positions, indices, params, mask, box, has_box, dist_eps)
        grad_params = (g * gp).astype(params.dtype)
    else:
        B = positions.shape[0]
        b_idx = indices.ndim > 2
        b_prm = params.ndim > 2
        b_msk = mask.ndim > 1
        b_box = box.ndim > 1

        def _step(b):
            gp_b = _angles_harmonic_param_grads(
                positions[b],
                indices[b] if b_idx else indices,
                params[b] if b_prm else params,
                mask[b] if b_msk else mask,
                box[b] if b_box else box,
                has_box,
                dist_eps,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (g_b * gp_b).astype(params.dtype)

        gp_all = jax.lax.map(_step, jnp.arange(B))
        grad_params = gp_all if b_prm else jnp.sum(gp_all, axis=0)

    return (
        grad_pos,
        jnp.zeros(shapes_meta[1].shape, dtype=shapes_meta[1].dtype),
        grad_params,
        jnp.zeros(shapes_meta[3].shape, dtype=shapes_meta[3].dtype),
        jnp.zeros(shapes_meta[4].shape, dtype=shapes_meta[4].dtype),
    )


_pallas_angles_harmonic_diff.defvjp(
    _pallas_angles_harmonic_fwd, _pallas_angles_harmonic_bwd, symbolic_zeros=True
)


def angles_harmonic_reference(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    *,
    has_box: bool = False,
    dist_eps: float = 1e-10,
) -> jax.Array:
    """Plain-JAX 3-body harmonic angle reference matching ``pallas_angles_harmonic``."""

    def _one(pos_1, idx_1, prm_1, msk_1, bx_1):
        N = pos_1.shape[0]
        dtype = pos_1.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        a_i, a_j, a_k = idx_1[:, 0].astype(jnp.int32), idx_1[:, 1].astype(jnp.int32), idx_1[:, 2].astype(jnp.int32)
        m_f = msk_1.astype(dtype)
        valid = (
            (m_f > zero)
            & (a_i >= 0)
            & (a_i < N)
            & (a_j >= 0)
            & (a_j < N)
            & (a_k >= 0)
            & (a_k < N)
        )
        m_eff = jnp.where(valid, m_f, zero)
        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
        safe_k = jnp.where(valid, jnp.clip(a_k, 0, max_atom), 0)
        v_ji = _min_image_host(pos_1[safe_i] - pos_1[safe_j], bx_1, has_box)
        v_jk = _min_image_host(pos_1[safe_k] - pos_1[safe_j], bx_1, has_box)
        d_ji_sq = jnp.sum(v_ji * v_ji, axis=-1)
        d_jk_sq = jnp.sum(v_jk * v_jk, axis=-1)
        if dist_eps > 0.0:
            d_ji = jnp.sqrt(d_ji_sq + jnp.asarray(dist_eps, dtype=dtype))
            d_jk = jnp.sqrt(d_jk_sq + jnp.asarray(dist_eps, dtype=dtype))
        else:
            d_ji = jnp.where(d_ji_sq > zero, jnp.sqrt(jnp.where(d_ji_sq > zero, d_ji_sq, one)), zero)
            d_jk = jnp.where(d_jk_sq > zero, jnp.sqrt(jnp.where(d_jk_sq > zero, d_jk_sq, one)), zero)
        cos_th = jnp.clip(
            jnp.sum(v_ji * v_jk, axis=-1) / (d_ji * d_jk + jnp.asarray(1e-8, dtype=dtype)),
            jnp.asarray(-0.999999, dtype=dtype),
            jnp.asarray(0.999999, dtype=dtype),
        )
        th = jnp.arccos(cos_th)
        diff = th - prm_1[:, 0].astype(dtype)
        return jnp.sum(jnp.asarray(0.5, dtype=dtype) * m_eff * prm_1[:, 1].astype(dtype) * diff * diff)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _one(
                positions[b],
                indices[b] if indices.ndim == 3 else indices,
                params[b] if params.ndim == 3 else params,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _one(positions, indices, params, mask, box)


def make_angles_harmonic_op(
    *,
    has_box: bool = False,
    dist_eps: float = 1e-10,
    angle_regularization_mode: str = "legacy_denom_eps",
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for 3-body harmonic angles.

    Signature of the returned op: ``(positions, indices, params, mask, box) -> energy``.
    """
    hb = bool(has_box)
    de = float(dist_eps)
    arm = str(angle_regularization_mode)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(angles_harmonic_reference, has_box=hb, dist_eps=de)

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(pos, idx, prm, msk, bx):
            return _pallas_angles_harmonic_fwd(pos, idx, prm, msk, bx, hb, de, interp, arm, cpg)

        def _bwd(res, ct):
            return _pallas_angles_harmonic_bwd(hb, de, interp, arm, cpg, res, ct)

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_angles_harmonic",
        context=context,
    )


def pallas_angles_harmonic(
    positions: jax.Array,
    indices: jax.Array | None,
    params: jax.Array | None,
    mask: jax.Array | None = None,
    box: jax.Array | None = None,
    *,
    dist_eps: float = 1e-10,
    angle_regularization_mode: str = "legacy_denom_eps",
    interpret: bool | None = None,
    compute_param_grads: bool | None = None,
) -> jax.Array:
    """1D-tiled (BLOCK_A=64) 3-body harmonic angle Pallas kernel."""
    if indices is None or params is None or indices.shape[-2] == 0:
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    idx_3col = jnp.asarray(indices[..., :, :3], dtype=jnp.int32)
    prm_arr = jnp.asarray(params, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(idx_3col.shape[:-1], dtype=positions.dtype)
        if mask is None
        else jnp.asarray(mask, dtype=positions.dtype)
    )
    box_arr, has_box = _normalize_box(box, positions)

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_angles_harmonic_op(
            has_box=bool(has_box),
            dist_eps=float(dist_eps),
            angle_regularization_mode=str(angle_regularization_mode),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(positions, idx_3col, prm_arr, msk_arr, box_arr)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_angles_harmonic_diff(
        positions,
        idx_3col,
        prm_arr,
        msk_arr,
        box_arr,
        bool(has_box),
        float(dist_eps),
        bool(interpret),
        str(angle_regularization_mode),
        cpg,
    )


# =============================================================================
# R3: 4-Body Torsion Helper & Kernel (pallas_torsions_4body)
# =============================================================================


def _torsion_angle_and_atom_grads_vec3(
    r_i: Vec3,
    r_j: Vec3,
    r_k: Vec3,
    r_l: Vec3,
    valid: jax.Array,
    box_xyz: Vec3,
    has_box: bool,
    b0_is_ji: bool,
) -> tuple[jax.Array, Vec3, Vec3, Vec3, Vec3]:
    """Compute torsion angle phi and analytical (dphi/dr_i, dphi/dr_j, dphi/dr_k, dphi/dr_l) as Vec3 via Bekker plane-normal cross-product identity (O3.1)."""
    dtype = r_i[0].dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    pi = jnp.asarray(jnp.pi, dtype=dtype)

    b0_ij = _min_image3(_sub3(r_i, r_j), box_xyz, has_box)
    b1 = _min_image3(_sub3(r_k, r_j), box_xyz, has_box)
    b2 = _min_image3(_sub3(r_l, r_k), box_xyz, has_box)

    n1 = _cross3(b0_ij, b1)
    n2 = _cross3(b1, b2)

    r_b1_sq = _dot3(b1, b1)
    if b0_is_ji:
        r_b1 = jnp.sqrt(r_b1_sq + jnp.asarray(1e-12, dtype=dtype))
    else:
        b1_sq_safe = jnp.where(r_b1_sq > zero, r_b1_sq, one)
        r_b1 = jnp.where(r_b1_sq > zero, jnp.sqrt(b1_sq_safe), zero)

    y_raw = (zero - r_b1) * _dot3(b0_ij, n2)
    x_raw = zero - _dot3(n1, n2)

    r_xy_sq_raw = x_raw * x_raw + y_raw * y_raw
    overlap = (~valid) | (r_xy_sq_raw < jnp.asarray(1e-24, dtype=dtype))
    if b0_is_ji:
        x_s = jnp.where(overlap, one, zero - x_raw)
        y_s = jnp.where(overlap, zero, zero - y_raw)
        phi = jnp.arctan2(y_s, x_s) - pi
    else:
        x_s = jnp.where(overlap, one, x_raw)
        y_s = jnp.where(overlap, zero, y_raw)
        phi = jnp.arctan2(y_s, x_s)

    n1_sq = jnp.maximum(_dot3(n1, n1), jnp.asarray(1e-24, dtype=dtype))
    n2_sq = jnp.maximum(_dot3(n2, n2), jnp.asarray(1e-24, dtype=dtype))
    r_b1_sq_safe = jnp.maximum(r_b1_sq, jnp.asarray(1e-12, dtype=dtype))

    scale_i = jnp.where(overlap, zero, r_b1 / n1_sq)
    scale_l = jnp.where(overlap, zero, r_b1 / n2_sq)
    dphi_dri = _scale3(scale_i, n1)
    dphi_drl = _scale3(scale_l, n2)

    s_phi = _add3(
        _scale3(_dot3(b0_ij, b1) / r_b1_sq_safe, dphi_dri),
        _scale3(_dot3(b2, b1) / r_b1_sq_safe, dphi_drl),
    )
    dphi_drj = _sub3(s_phi, dphi_dri)
    dphi_drk = _sub3(_scale3(zero - one, s_phi), dphi_drl)

    return phi, dphi_dri, dphi_drj, dphi_drk, dphi_drl


def _pallas_torsions_4body_single(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    b0_is_ji: bool,
    is_harmonic: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    D_real = indices.shape[0]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N

    n_blocks = max(1, (D_real + _BLOCK_D - 1) // _BLOCK_D)
    D_pad = n_blocks * _BLOCK_D
    pad_d = D_pad - D_real

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    idx_i = indices[:, 0].astype(jnp.int32)
    idx_j = indices[:, 1].astype(jnp.int32)
    idx_k = indices[:, 2].astype(jnp.int32)
    idx_l = indices[:, 3].astype(jnp.int32)
    p0 = params[:, 0].astype(dtype)
    p1 = params[:, 1].astype(dtype)
    p2 = params[:, 2].astype(dtype) if params.shape[-1] >= 3 else jnp.zeros_like(p0)
    mask_f = mask.astype(dtype)

    if pad_d > 0:
        idx_i = jnp.pad(idx_i, (0, pad_d), constant_values=-1)
        idx_j = jnp.pad(idx_j, (0, pad_d), constant_values=-1)
        idx_k = jnp.pad(idx_k, (0, pad_d), constant_values=-1)
        idx_l = jnp.pad(idx_l, (0, pad_d), constant_values=-1)
        p0 = jnp.pad(p0, (0, pad_d))
        p1 = jnp.pad(p1, (0, pad_d))
        p2 = jnp.pad(p2, (0, pad_d))
        mask_f = jnp.pad(mask_f, (0, pad_d))

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    box_vec = jnp.asarray(box.astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _torsions_kernel(
        px_ref,
        py_ref,
        pz_ref,
        idx_i_ref,
        idx_j_ref,
        idx_k_ref,
        idx_l_ref,
        p0_ref,
        p1_ref,
        p2_ref,
        mask_ref,
        box_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        offs = pid * _BLOCK_D + jnp.arange(_BLOCK_D, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)

        a_i = _pl_load(idx_i_ref, (offs,))
        a_j = _pl_load(idx_j_ref, (offs,))
        a_k = _pl_load(idx_k_ref, (offs,))
        a_l = _pl_load(idx_l_ref, (offs,))
        m_val = _pl_load(mask_ref, (offs,))
        valid = (
            (offs < D_real)
            & (m_val > zero)
            & (a_i >= 0)
            & (a_i < N)
            & (a_j >= 0)
            & (a_j < N)
            & (a_k >= 0)
            & (a_k < N)
            & (a_l >= 0)
            & (a_l < N)
        )
        m_eff = jnp.where(valid, m_val, zero)

        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
        safe_k = jnp.where(valid, jnp.clip(a_k, 0, max_atom), 0)
        safe_l = jnp.where(valid, jnp.clip(a_l, 0, max_atom), 0)

        r_i = (_pl_load(px_ref, (safe_i,)), _pl_load(py_ref, (safe_i,)), _pl_load(pz_ref, (safe_i,)))
        r_j = (_pl_load(px_ref, (safe_j,)), _pl_load(py_ref, (safe_j,)), _pl_load(pz_ref, (safe_j,)))
        r_k = (_pl_load(px_ref, (safe_k,)), _pl_load(py_ref, (safe_k,)), _pl_load(pz_ref, (safe_k,)))
        r_l = (_pl_load(px_ref, (safe_l,)), _pl_load(py_ref, (safe_l,)), _pl_load(pz_ref, (safe_l,)))

        phi, dphi_dri, dphi_drj, dphi_drk, dphi_drl = _torsion_angle_and_atom_grads_vec3(
            r_i, r_j, r_k, r_l, valid, _load_box6(box_ref), has_box, b0_is_ji
        )

        prm0 = _pl_load(p0_ref, (offs,))
        prm1 = _pl_load(p1_ref, (offs,))
        prm2 = _pl_load(p2_ref, (offs,))

        if is_harmonic:
            # prm0 = k, prm1 = phi0
            diff_raw = phi - prm1 + pi
            dphi = diff_raw - two_pi * jnp.floor(diff_raw / two_pi) - pi
            e_d = m_eff * prm0 * dphi * dphi
            g_phi = two * m_eff * prm0 * dphi
        else:
            # prm0 = n, prm1 = phase, prm2 = k
            arg = prm0 * phi - prm1
            e_d = m_eff * prm2 * (one + jnp.cos(arg))
            g_phi = (zero - m_eff) * prm2 * prm0 * jnp.sin(arg)

        _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_d)[None])

        neg_g_phi = jnp.where(valid, zero - g_phi, zero)
        f_i = _scale3(neg_g_phi, dphi_dri)
        f_j = _scale3(neg_g_phi, dphi_drj)
        f_k = _scale3(neg_g_phi, dphi_drk)
        f_l = _scale3(neg_g_phi, dphi_drl)

        _pl_atomic_add_1d(fx_out_ref, safe_i, f_i[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_i, f_i[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_i, f_i[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_j, f_j[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_j, f_j[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_j, f_j[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_k, f_k[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_k, f_k[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_k, f_k[2], valid, interpret)
        _pl_atomic_add_1d(fx_out_ref, safe_l, f_l[0], valid, interpret)
        _pl_atomic_add_1d(fy_out_ref, safe_l, f_l[1], valid, interpret)
        _pl_atomic_add_1d(fz_out_ref, safe_l, f_l[2], valid, interpret)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(15)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((n_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _torsions_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        input_output_aliases={12: 1, 13: 2, 14: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_torsions_4body",
    )(
        px,
        py,
        pz,
        idx_i,
        idx_j,
        idx_k,
        idx_l,
        p0,
        p1,
        p2,
        mask_f,
        box_vec,
        zero_f,
        zero_f,
        zero_f,
    )
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _torsions_4body_param_grads(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    b0_is_ji: bool,
    is_harmonic: bool,
) -> jax.Array:
    N = positions.shape[0]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    pi = jnp.asarray(jnp.pi, dtype=dtype)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)

    a_i = indices[:, 0].astype(jnp.int32)
    a_j = indices[:, 1].astype(jnp.int32)
    a_k = indices[:, 2].astype(jnp.int32)
    a_l = indices[:, 3].astype(jnp.int32)
    m_f = mask.astype(dtype)
    valid = (
        (m_f > zero)
        & (a_i >= 0)
        & (a_i < N)
        & (a_j >= 0)
        & (a_j < N)
        & (a_k >= 0)
        & (a_k < N)
        & (a_l >= 0)
        & (a_l < N)
    )
    m_eff = jnp.where(valid, m_f, zero)
    max_atom = max(N - 1, 0)
    safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
    safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
    safe_k = jnp.where(valid, jnp.clip(a_k, 0, max_atom), 0)
    safe_l = jnp.where(valid, jnp.clip(a_l, 0, max_atom), 0)

    r_i = (positions[safe_i, 0], positions[safe_i, 1], positions[safe_i, 2])
    r_j = (positions[safe_j, 0], positions[safe_j, 1], positions[safe_j, 2])
    r_k = (positions[safe_k, 0], positions[safe_k, 1], positions[safe_k, 2])
    r_l = (positions[safe_l, 0], positions[safe_l, 1], positions[safe_l, 2])
    phi, _, _, _, _ = _torsion_angle_and_atom_grads_vec3(
        r_i, r_j, r_k, r_l, valid, box, has_box, b0_is_ji
    )

    P = params.shape[-1]
    if is_harmonic:
        k_d = params[:, 0].astype(dtype)
        phi0 = params[:, 1].astype(dtype)
        diff_raw = phi - phi0 + pi
        dphi = diff_raw - two_pi * jnp.floor(diff_raw / two_pi) - pi
        grad_k = m_eff * dphi * dphi
        grad_phi0 = -2.0 * m_eff * k_d * dphi
        cols = [grad_k, grad_phi0]
    else:
        n_d = params[:, 0].astype(dtype)
        phase_d = params[:, 1].astype(dtype)
        k_d = params[:, 2].astype(dtype)
        arg = n_d * phi - phase_d
        sin_arg = jnp.sin(arg)
        grad_n = -m_eff * k_d * phi * sin_arg
        grad_phase = m_eff * k_d * sin_arg
        grad_k = m_eff * (one + jnp.cos(arg))
        cols = [grad_n, grad_phase, grad_k]

    while len(cols) < P:
        cols.append(jnp.zeros_like(cols[0]))
    return jnp.stack(cols[:P], axis=-1).astype(params.dtype)


def _pallas_torsions_4body_vmappable(
    positions, indices, params, mask, box, has_box, b0_is_ji, is_harmonic, interpret
):
    hb = bool(has_box)
    b0_ji = bool(b0_is_ji)
    harm = bool(is_harmonic)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, idx, prm, msk, bx):
        return _pallas_torsions_4body_single(pos, idx, prm, msk, bx, hb, b0_ji, harm, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, idx, prm, msk, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                idx[b_idx] if in_batched[1] else idx,
                prm[b_idx] if in_batched[2] else prm,
                msk[b_idx] if in_batched[3] else msk,
                bx[b_idx] if in_batched[4] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                indices[b] if indices.ndim == 3 else indices,
                params[b] if params.ndim == 3 else params,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _vmappable(positions, indices, params, mask, box)


@functools.partial(custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _pallas_torsions_4body_diff(
    positions,
    indices,
    params,
    mask,
    box,
    has_box,
    b0_is_ji,
    is_harmonic,
    interpret,
    compute_param_grads=None,
):
    energy, _ = _pallas_torsions_4body_vmappable(
        positions, indices, params, mask, box, has_box, b0_is_ji, is_harmonic, interpret
    )
    return energy


def _pallas_torsions_4body_fwd(
    positions,
    indices,
    params,
    mask,
    box,
    has_box,
    b0_is_ji,
    is_harmonic,
    interpret,
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    idx_v = _unwrap_primal(indices)
    prm_v = _unwrap_primal(params)
    msk_v = _unwrap_primal(mask)
    bx_v = _unwrap_primal(box)

    energy, forces = _pallas_torsions_4body_vmappable(
        pos_v, idx_v, prm_v, msk_v, bx_v, has_box, b0_is_ji, is_harmonic, interpret
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    any_param_perturbed = (
        bool(compute_param_grads)
        if compute_param_grads is not None
        else _is_perturbed(params, default_pert)
    )
    shapes_meta = _make_shapes_meta(pos_v, idx_v, prm_v, msk_v, bx_v)
    param_res = (pos_v, idx_v, prm_v, msk_v, bx_v) if any_param_perturbed else None
    return energy, (forces, shapes_meta, param_res)


def _pallas_torsions_4body_bwd(
    has_box, b0_is_ji, is_harmonic, interpret, compute_param_grads, res, g
):
    del interpret, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    positions, indices, params, mask, box = param_res
    if positions.ndim == 2:
        gp = _torsions_4body_param_grads(
            positions, indices, params, mask, box, has_box, b0_is_ji, is_harmonic
        )
        grad_params = (g * gp).astype(params.dtype)
    else:
        B = positions.shape[0]
        b_idx = indices.ndim > 2
        b_prm = params.ndim > 2
        b_msk = mask.ndim > 1
        b_box = box.ndim > 1

        def _step(b):
            gp_b = _torsions_4body_param_grads(
                positions[b],
                indices[b] if b_idx else indices,
                params[b] if b_prm else params,
                mask[b] if b_msk else mask,
                box[b] if b_box else box,
                has_box,
                b0_is_ji,
                is_harmonic,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (g_b * gp_b).astype(params.dtype)

        gp_all = jax.lax.map(_step, jnp.arange(B))
        grad_params = gp_all if b_prm else jnp.sum(gp_all, axis=0)

    return (
        grad_pos,
        jnp.zeros(shapes_meta[1].shape, dtype=shapes_meta[1].dtype),
        grad_params,
        jnp.zeros(shapes_meta[3].shape, dtype=shapes_meta[3].dtype),
        jnp.zeros(shapes_meta[4].shape, dtype=shapes_meta[4].dtype),
    )


_pallas_torsions_4body_diff.defvjp(
    _pallas_torsions_4body_fwd, _pallas_torsions_4body_bwd, symbolic_zeros=True
)


def torsions_4body_reference(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    *,
    has_box: bool = False,
    b0_is_ji: bool = True,
    is_harmonic: bool = False,
) -> jax.Array:
    """Plain-JAX 4-body torsion reference matching ``pallas_torsions_4body``."""

    def _one(pos_1, idx_1, prm_1, msk_1, bx_1):
        N = pos_1.shape[0]
        dtype = pos_1.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
        a_i, a_j, a_k, a_l = (
            idx_1[:, 0].astype(jnp.int32),
            idx_1[:, 1].astype(jnp.int32),
            idx_1[:, 2].astype(jnp.int32),
            idx_1[:, 3].astype(jnp.int32),
        )
        m_f = msk_1.astype(dtype)
        valid = (
            (m_f > zero)
            & (a_i >= 0)
            & (a_i < N)
            & (a_j >= 0)
            & (a_j < N)
            & (a_k >= 0)
            & (a_k < N)
            & (a_l >= 0)
            & (a_l < N)
        )
        m_eff = jnp.where(valid, m_f, zero)
        max_atom = max(N - 1, 0)
        safe_i = jnp.where(valid, jnp.clip(a_i, 0, max_atom), 0)
        safe_j = jnp.where(valid, jnp.clip(a_j, 0, max_atom), 0)
        safe_k = jnp.where(valid, jnp.clip(a_k, 0, max_atom), 0)
        safe_l = jnp.where(valid, jnp.clip(a_l, 0, max_atom), 0)
        r_i = (pos_1[safe_i, 0], pos_1[safe_i, 1], pos_1[safe_i, 2])
        r_j = (pos_1[safe_j, 0], pos_1[safe_j, 1], pos_1[safe_j, 2])
        r_k = (pos_1[safe_k, 0], pos_1[safe_k, 1], pos_1[safe_k, 2])
        r_l = (pos_1[safe_l, 0], pos_1[safe_l, 1], pos_1[safe_l, 2])
        phi, _, _, _, _ = _torsion_angle_and_atom_grads_vec3(
            r_i, r_j, r_k, r_l, valid, bx_1, has_box, b0_is_ji
        )
        if is_harmonic or prm_1.shape[-1] < 3:
            diff_raw = phi - prm_1[:, 1].astype(dtype) + pi
            dphi = diff_raw - two_pi * jnp.floor(diff_raw / two_pi) - pi
            return jnp.sum(m_eff * prm_1[:, 0].astype(dtype) * dphi * dphi)
        return jnp.sum(
            m_eff
            * prm_1[:, 2].astype(dtype)
            * (one + jnp.cos(prm_1[:, 0].astype(dtype) * phi - prm_1[:, 1].astype(dtype)))
        )

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _one(
                positions[b],
                indices[b] if indices.ndim == 3 else indices,
                params[b] if params.ndim == 3 else params,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _one(positions, indices, params, mask, box)


def make_torsions_4body_op(
    *,
    has_box: bool = False,
    b0_is_ji: bool = True,
    is_harmonic: bool = False,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for 4-body torsions (dihedrals & periodic/harmonic impropers).

    Signature of the returned op: ``(positions, indices, params, mask, box) -> energy``.
    """
    hb = bool(has_box)
    b0_ji = bool(b0_is_ji)
    harm = bool(is_harmonic)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(
        torsions_4body_reference, has_box=hb, b0_is_ji=b0_ji, is_harmonic=harm
    )

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(pos, idx, prm, msk, bx):
            return _pallas_torsions_4body_fwd(
                pos, idx, prm, msk, bx, hb, b0_ji, harm, interp, cpg
            )

        def _bwd(res, ct):
            return _pallas_torsions_4body_bwd(hb, b0_ji, harm, interp, cpg, res, ct)

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_torsions_4body",
        context=context,
    )


def pallas_torsions_4body(
    positions: jax.Array,
    indices: jax.Array | None,
    params: jax.Array | None,
    mask: jax.Array | None = None,
    box: jax.Array | None = None,
    *,
    b0_is_ji: bool = True,
    is_harmonic: bool = False,
    interpret: bool | None = None,
    compute_param_grads: bool | None = None,
) -> jax.Array:
    """1D-tiled (BLOCK_D=64) 4-body torsion Pallas kernel (dihedrals & periodic/harmonic impropers)."""
    if indices is None or params is None or indices.shape[-2] == 0:
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    idx_4col = jnp.asarray(indices[..., :, :4], dtype=jnp.int32)
    prm_arr = jnp.asarray(params, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(idx_4col.shape[:-1], dtype=positions.dtype)
        if mask is None
        else jnp.asarray(mask, dtype=positions.dtype)
    )

    # Flatten multi-term (..., N_dih, T, P) params along the torsion axis
    n_dih = idx_4col.shape[-2]
    is_multi_term = (prm_arr.ndim == 4) or (
        prm_arr.ndim == 3
        and not (
            positions.ndim == 3
            and prm_arr.shape[0] == positions.shape[0]
            and prm_arr.shape[1] == n_dih
        )
    )
    if is_multi_term:
        T = prm_arr.shape[-2]
        P = prm_arr.shape[-1]
        idx_4col = jnp.repeat(idx_4col, T, axis=-2)
        prm_arr = prm_arr.reshape(*prm_arr.shape[:-3], prm_arr.shape[-3] * T, P)
        msk_arr = jnp.repeat(msk_arr, T, axis=-1)

    box_arr, has_box = _normalize_box(box, positions)
    eff_harm = bool(is_harmonic) or (prm_arr.shape[-1] < 3)

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_torsions_4body_op(
            has_box=bool(has_box),
            b0_is_ji=bool(b0_is_ji),
            is_harmonic=eff_harm,
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(positions, idx_4col, prm_arr, msk_arr, box_arr)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_torsions_4body_diff(
        positions,
        idx_4col,
        prm_arr,
        msk_arr,
        box_arr,
        bool(has_box),
        bool(b0_is_ji),
        eff_harm,
        bool(interpret),
        cpg,
    )


# =============================================================================
# R4: 5-Body CMAP 2D Bicubic Spline Kernel (pallas_cmap_torsions)
# =============================================================================


def _eval_cmap_horner_cell(
    coeffs_flat_ref,
    base_idx: jax.Array,
    u: jax.Array,
    v: jax.Array,
    dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 2D bicubic spline E(u, v), dE/du, dE/dv via Horner 4-tuple rule (O3.3)."""
    two = jnp.asarray(2.0, dtype=dtype)
    three = jnp.asarray(3.0, dtype=dtype)
    a_vals = []
    da_vals = []
    for p in range(4):
        c_p0 = _pl_load(coeffs_flat_ref, (base_idx + (4 * p + 0),))
        c_p1 = _pl_load(coeffs_flat_ref, (base_idx + (4 * p + 1),))
        c_p2 = _pl_load(coeffs_flat_ref, (base_idx + (4 * p + 2),))
        c_p3 = _pl_load(coeffs_flat_ref, (base_idx + (4 * p + 3),))
        a_p = ((c_p3 * v + c_p2) * v + c_p1) * v + c_p0
        da_p = (three * c_p3 * v + two * c_p2) * v + c_p1
        a_vals.append(a_p)
        da_vals.append(da_p)
    a0, a1, a2, a3 = a_vals
    da0, da1, da2, da3 = da_vals
    e_poly = ((a3 * u + a2) * u + a1) * u + a0
    de_du = (three * a3 * u + two * a2) * u + a1
    de_dv = ((da3 * u + da2) * u + da1) * u + da0
    return e_poly, de_du, de_dv


def _pallas_cmap_torsions_single(
    positions: jax.Array,
    cmap_torsions: jax.Array,
    map_indices: jax.Array,
    cmap_coeffs: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    first_is_psi: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    C_real = cmap_torsions.shape[0]
    N_maps = cmap_coeffs.shape[0]
    G = cmap_coeffs.shape[1]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N

    n_blocks = max(1, (C_real + _BLOCK_C - 1) // _BLOCK_C)
    C_pad = n_blocks * _BLOCK_C
    pad_c = C_pad - C_real

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    a0 = cmap_torsions[:, 0].astype(jnp.int32)
    a1 = cmap_torsions[:, 1].astype(jnp.int32)
    a2 = cmap_torsions[:, 2].astype(jnp.int32)
    a3 = cmap_torsions[:, 3].astype(jnp.int32)
    a4 = cmap_torsions[:, 4].astype(jnp.int32)
    m_idx = map_indices.astype(jnp.int32)
    mask_f = mask.astype(dtype)

    if pad_c > 0:
        a0 = jnp.pad(a0, (0, pad_c), constant_values=-1)
        a1 = jnp.pad(a1, (0, pad_c), constant_values=-1)
        a2 = jnp.pad(a2, (0, pad_c), constant_values=-1)
        a3 = jnp.pad(a3, (0, pad_c), constant_values=-1)
        a4 = jnp.pad(a4, (0, pad_c), constant_values=-1)
        m_idx = jnp.pad(m_idx, (0, pad_c), constant_values=0)
        mask_f = jnp.pad(mask_f, (0, pad_c))

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    coeffs_flat = jnp.asarray(cmap_coeffs.astype(dtype).reshape(-1))
    box_vec = jnp.asarray(box.astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _cmap_kernel(
        px_ref,
        py_ref,
        pz_ref,
        a0_ref,
        a1_ref,
        a2_ref,
        a3_ref,
        a4_ref,
        map_idx_ref,
        coeffs_flat_ref,
        mask_ref,
        box_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        offs = pid * _BLOCK_C + jnp.arange(_BLOCK_C, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
        delta = jnp.asarray(2.0 * jnp.pi / G, dtype=dtype)
        inv_delta = one / delta

        i0 = _pl_load(a0_ref, (offs,))
        i1 = _pl_load(a1_ref, (offs,))
        i2 = _pl_load(a2_ref, (offs,))
        i3 = _pl_load(a3_ref, (offs,))
        i4 = _pl_load(a4_ref, (offs,))
        m_val = _pl_load(mask_ref, (offs,))
        valid = (
            (offs < C_real)
            & (m_val > zero)
            & (i0 >= 0)
            & (i0 < N)
            & (i1 >= 0)
            & (i1 < N)
            & (i2 >= 0)
            & (i2 < N)
            & (i3 >= 0)
            & (i3 < N)
            & (i4 >= 0)
            & (i4 < N)
        )
        m_eff = jnp.where(valid, m_val, zero)

        max_atom = max(N - 1, 0)
        s0 = jnp.where(valid, jnp.clip(i0, 0, max_atom), 0)
        s1 = jnp.where(valid, jnp.clip(i1, 0, max_atom), 0)
        s2 = jnp.where(valid, jnp.clip(i2, 0, max_atom), 0)
        s3 = jnp.where(valid, jnp.clip(i3, 0, max_atom), 0)
        s4 = jnp.where(valid, jnp.clip(i4, 0, max_atom), 0)

        r0 = (_pl_load(px_ref, (s0,)), _pl_load(py_ref, (s0,)), _pl_load(pz_ref, (s0,)))
        r1 = (_pl_load(px_ref, (s1,)), _pl_load(py_ref, (s1,)), _pl_load(pz_ref, (s1,)))
        r2 = (_pl_load(px_ref, (s2,)), _pl_load(py_ref, (s2,)), _pl_load(pz_ref, (s2,)))
        r3 = (_pl_load(px_ref, (s3,)), _pl_load(py_ref, (s3,)), _pl_load(pz_ref, (s3,)))
        r4 = (_pl_load(px_ref, (s4,)), _pl_load(py_ref, (s4,)), _pl_load(pz_ref, (s4,)))

        box_xyz = _load_box6(box_ref)

        phi, dphi_0, dphi_1, dphi_2, dphi_3 = _torsion_angle_and_atom_grads_vec3(
            r0, r1, r2, r3, valid, box_xyz, has_box, False
        )
        psi, dpsi_1, dpsi_2, dpsi_3, dpsi_4 = _torsion_angle_and_atom_grads_vec3(
            r1, r2, r3, r4, valid, box_xyz, has_box, False
        )

        if first_is_psi:
            theta1, theta2 = psi, phi
        else:
            theta1, theta2 = phi, psi

        th1_shift = theta1 + two_pi
        th1_norm = th1_shift - two_pi * jnp.floor(th1_shift / two_pi)
        th2_shift = theta2 + two_pi
        th2_norm = th2_shift - two_pi * jnp.floor(th2_shift / two_pi)

        q1 = th1_norm * inv_delta
        q2 = th2_norm * inv_delta
        g1_raw = jnp.floor(q1).astype(jnp.int32)
        g2_raw = jnp.floor(q2).astype(jnp.int32)
        g1 = g1_raw % G
        g2 = g2_raw % G
        u = q1 - g1_raw.astype(dtype)
        v = q2 - g2_raw.astype(dtype)

        m_raw = _pl_load(map_idx_ref, (offs,))
        m_safe = jnp.where(valid, jnp.clip(m_raw, 0, max(N_maps - 1, 0)), 0)
        base_idx = ((m_safe * G + g1) * G + g2) * 16

        e_poly, de_du, de_dv = _eval_cmap_horner_cell(
            coeffs_flat_ref, base_idx, u, v, dtype
        )

        e_c = m_eff * e_poly
        _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), jnp.sum(e_c)[None])

        de_dtheta1 = m_eff * de_du * inv_delta
        de_dtheta2 = m_eff * de_dv * inv_delta
        if first_is_psi:
            de_dpsi, de_dphi = de_dtheta1, de_dtheta2
        else:
            de_dphi, de_dpsi = de_dtheta1, de_dtheta2

        neg_dphi = jnp.where(valid, zero - de_dphi, zero)
        neg_dpsi = jnp.where(valid, zero - de_dpsi, zero)

        f0 = _scale3(neg_dphi, dphi_0)
        f1 = _add3(_scale3(neg_dphi, dphi_1), _scale3(neg_dpsi, dpsi_1))
        f2 = _add3(_scale3(neg_dphi, dphi_2), _scale3(neg_dpsi, dpsi_2))
        f3 = _add3(_scale3(neg_dphi, dphi_3), _scale3(neg_dpsi, dpsi_3))
        f4 = _scale3(neg_dpsi, dpsi_4)

        for s_atom, f_atom in ((s0, f0), (s1, f1), (s2, f2), (s3, f3), (s4, f4)):
            _pl_atomic_add_1d(fx_out_ref, s_atom, f_atom[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, s_atom, f_atom[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, s_atom, f_atom[2], valid, interpret)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(15)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((n_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _cmap_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        input_output_aliases={12: 1, 13: 2, 14: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_cmap_torsions",
    )(
        px,
        py,
        pz,
        a0,
        a1,
        a2,
        a3,
        a4,
        m_idx,
        coeffs_flat,
        mask_f,
        box_vec,
        zero_f,
        zero_f,
        zero_f,
    )
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _cmap_coeffs_grad(
    positions: jax.Array,
    cmap_torsions: jax.Array,
    map_indices: jax.Array,
    cmap_coeffs: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    has_box: bool,
    first_is_psi: bool,
) -> jax.Array:
    N = positions.shape[0]
    N_maps, G = cmap_coeffs.shape[0], cmap_coeffs.shape[1]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    i0, i1, i2, i3, i4 = (
        cmap_torsions[:, 0].astype(jnp.int32),
        cmap_torsions[:, 1].astype(jnp.int32),
        cmap_torsions[:, 2].astype(jnp.int32),
        cmap_torsions[:, 3].astype(jnp.int32),
        cmap_torsions[:, 4].astype(jnp.int32),
    )
    m_f = mask.astype(dtype)
    valid = (
        (m_f > zero)
        & (i0 >= 0)
        & (i0 < N)
        & (i1 >= 0)
        & (i1 < N)
        & (i2 >= 0)
        & (i2 < N)
        & (i3 >= 0)
        & (i3 < N)
        & (i4 >= 0)
        & (i4 < N)
    )
    m_eff = jnp.where(valid, m_f, zero)
    max_atom = max(N - 1, 0)
    s0 = jnp.where(valid, jnp.clip(i0, 0, max_atom), 0)
    s1 = jnp.where(valid, jnp.clip(i1, 0, max_atom), 0)
    s2 = jnp.where(valid, jnp.clip(i2, 0, max_atom), 0)
    s3 = jnp.where(valid, jnp.clip(i3, 0, max_atom), 0)
    s4 = jnp.where(valid, jnp.clip(i4, 0, max_atom), 0)

    r0 = (positions[s0, 0], positions[s0, 1], positions[s0, 2])
    r1 = (positions[s1, 0], positions[s1, 1], positions[s1, 2])
    r2 = (positions[s2, 0], positions[s2, 1], positions[s2, 2])
    r3 = (positions[s3, 0], positions[s3, 1], positions[s3, 2])
    r4 = (positions[s4, 0], positions[s4, 1], positions[s4, 2])
    box_xyz = box
    phi, _, _, _, _ = _torsion_angle_and_atom_grads_vec3(r0, r1, r2, r3, valid, box_xyz, has_box, False)
    psi, _, _, _, _ = _torsion_angle_and_atom_grads_vec3(r1, r2, r3, r4, valid, box_xyz, has_box, False)

    theta1, theta2 = (psi, phi) if first_is_psi else (phi, psi)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
    delta = jnp.asarray(2.0 * jnp.pi / G, dtype=dtype)
    th1_norm = jnp.mod(theta1 + two_pi, two_pi)
    th2_norm = jnp.mod(theta2 + two_pi, two_pi)
    q1 = th1_norm / delta
    q2 = th2_norm / delta
    g1_raw = jnp.floor(q1).astype(jnp.int32)
    g2_raw = jnp.floor(q2).astype(jnp.int32)
    g1 = g1_raw % G
    g2 = g2_raw % G
    u = q1 - g1_raw.astype(dtype)
    v = q2 - g2_raw.astype(dtype)

    u_vec = jnp.stack([jnp.ones_like(u), u, u * u, u * u * u], axis=-1)
    v_vec = jnp.stack([jnp.ones_like(v), v, v * v, v * v * v], axis=-1)
    uv_outer = (m_eff[:, None, None] * (u_vec[:, :, None] * v_vec[:, None, :])).reshape(-1, 16)
    m_safe = jnp.where(valid, jnp.clip(map_indices.astype(jnp.int32), 0, max(N_maps - 1, 0)), 0)
    grad_c = jnp.zeros_like(cmap_coeffs, dtype=dtype).at[m_safe, g1, g2].add(uv_outer)
    return grad_c.astype(cmap_coeffs.dtype)


def _pallas_cmap_torsions_vmappable(
    positions,
    cmap_torsions,
    map_indices,
    cmap_coeffs,
    mask,
    box,
    has_box,
    first_is_psi,
    interpret,
):
    hb = bool(has_box)
    fpsi = bool(first_is_psi)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, tors, midx, coeffs, msk, bx):
        return _pallas_cmap_torsions_single(pos, tors, midx, coeffs, msk, bx, hb, fpsi, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, tors, midx, coeffs, msk, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                tors[b_idx] if in_batched[1] else tors,
                midx[b_idx] if in_batched[2] else midx,
                coeffs[b_idx] if in_batched[3] else coeffs,
                msk[b_idx] if in_batched[4] else msk,
                bx[b_idx] if in_batched[5] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                cmap_torsions[b] if cmap_torsions.ndim == 3 else cmap_torsions,
                map_indices[b] if map_indices.ndim == 2 else map_indices,
                cmap_coeffs[b] if cmap_coeffs.ndim == 5 else cmap_coeffs,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _vmappable(positions, cmap_torsions, map_indices, cmap_coeffs, mask, box)


@functools.partial(custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def _pallas_cmap_torsions_diff(
    positions,
    cmap_torsions,
    map_indices,
    cmap_coeffs,
    mask,
    box,
    has_box,
    first_is_psi,
    interpret,
    compute_param_grads=None,
):
    energy, _ = _pallas_cmap_torsions_vmappable(
        positions,
        cmap_torsions,
        map_indices,
        cmap_coeffs,
        mask,
        box,
        has_box,
        first_is_psi,
        interpret,
    )
    return energy


def _pallas_cmap_torsions_fwd(
    positions,
    cmap_torsions,
    map_indices,
    cmap_coeffs,
    mask,
    box,
    has_box,
    first_is_psi,
    interpret,
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    tors_v = _unwrap_primal(cmap_torsions)
    midx_v = _unwrap_primal(map_indices)
    coeffs_v = _unwrap_primal(cmap_coeffs)
    msk_v = _unwrap_primal(mask)
    bx_v = _unwrap_primal(box)

    energy, forces = _pallas_cmap_torsions_vmappable(
        pos_v,
        tors_v,
        midx_v,
        coeffs_v,
        msk_v,
        bx_v,
        has_box,
        first_is_psi,
        interpret,
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    any_param_perturbed = (
        bool(compute_param_grads)
        if compute_param_grads is not None
        else _is_perturbed(cmap_coeffs, default_pert)
    )
    shapes_meta = _make_shapes_meta(pos_v, tors_v, midx_v, coeffs_v, msk_v, bx_v)
    param_res = (
        (pos_v, tors_v, midx_v, coeffs_v, msk_v, bx_v)
        if any_param_perturbed
        else None
    )
    return energy, (forces, shapes_meta, param_res)


def _pallas_cmap_torsions_bwd(
    has_box, first_is_psi, interpret, compute_param_grads, res, g
):
    del interpret, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    positions, cmap_torsions, map_indices, cmap_coeffs, mask, box = param_res
    if positions.ndim == 2:
        gc = _cmap_coeffs_grad(
            positions, cmap_torsions, map_indices, cmap_coeffs, mask, box, has_box, first_is_psi
        )
        grad_coeffs = (g * gc).astype(cmap_coeffs.dtype)
    else:
        B = positions.shape[0]
        b_tors = cmap_torsions.ndim > 2
        b_midx = map_indices.ndim > 1
        b_coeffs = cmap_coeffs.ndim > 4
        b_msk = mask.ndim > 1
        b_box = box.ndim > 1

        def _step(b):
            gc_b = _cmap_coeffs_grad(
                positions[b],
                cmap_torsions[b] if b_tors else cmap_torsions,
                map_indices[b] if b_midx else map_indices,
                cmap_coeffs[b] if b_coeffs else cmap_coeffs,
                mask[b] if b_msk else mask,
                box[b] if b_box else box,
                has_box,
                first_is_psi,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (g_b * gc_b).astype(cmap_coeffs.dtype)

        gc_all = jax.lax.map(_step, jnp.arange(B))
        grad_coeffs = gc_all if b_coeffs else jnp.sum(gc_all, axis=0)

    return (
        grad_pos,
        jnp.zeros(shapes_meta[1].shape, dtype=shapes_meta[1].dtype),
        jnp.zeros(shapes_meta[2].shape, dtype=shapes_meta[2].dtype),
        grad_coeffs,
        jnp.zeros(shapes_meta[4].shape, dtype=shapes_meta[4].dtype),
        jnp.zeros(shapes_meta[5].shape, dtype=shapes_meta[5].dtype),
    )


_pallas_cmap_torsions_diff.defvjp(
    _pallas_cmap_torsions_fwd, _pallas_cmap_torsions_bwd, symbolic_zeros=True
)


def cmap_torsions_reference(
    positions: jax.Array,
    cmap_torsions: jax.Array,
    map_indices: jax.Array,
    cmap_coeffs: jax.Array,
    mask: jax.Array,
    box: jax.Array,
    *,
    has_box: bool = False,
    first_is_psi: bool = False,
) -> jax.Array:
    """Plain-JAX 5-body CMAP bicubic spline reference matching ``pallas_cmap_torsions``."""

    def _one(pos_1, t_1, mi_1, c_1, m_1, bx_1):
        N = pos_1.shape[0]
        dtype = pos_1.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        m_f = m_1.astype(dtype)
        valid = m_f > zero
        for col in range(5):
            valid = valid & (t_1[:, col] >= 0) & (t_1[:, col] < N)
        m_eff = jnp.where(valid, m_f, zero)
        max_atom = max(N - 1, 0)
        t_safe = jnp.where(valid[:, None], jnp.clip(t_1, 0, max_atom), 0)

        def _angle4(i0, i1, i2, i3):
            r0 = (pos_1[i0, 0], pos_1[i0, 1], pos_1[i0, 2])
            r1 = (pos_1[i1, 0], pos_1[i1, 1], pos_1[i1, 2])
            r2 = (pos_1[i2, 0], pos_1[i2, 1], pos_1[i2, 2])
            r3 = (pos_1[i3, 0], pos_1[i3, 1], pos_1[i3, 2])
            ang, _, _, _, _ = _torsion_angle_and_atom_grads_vec3(
                r0, r1, r2, r3, valid, bx_1, has_box, False
            )
            return ang

        phi = _angle4(t_safe[:, 0], t_safe[:, 1], t_safe[:, 2], t_safe[:, 3])
        psi = _angle4(t_safe[:, 1], t_safe[:, 2], t_safe[:, 3], t_safe[:, 4])
        th1, th2 = (psi, phi) if first_is_psi else (phi, psi)
        e_per = compute_cmap_energies(th1, th2, mi_1, c_1.astype(dtype))
        return jnp.sum(e_per * m_eff).astype(dtype)

    if positions.ndim == 3:
        B = positions.shape[0]
        return jax.vmap(
            lambda b: _one(
                positions[b],
                cmap_torsions[b] if cmap_torsions.ndim == 3 else cmap_torsions,
                map_indices[b] if map_indices.ndim == 2 else map_indices,
                cmap_coeffs[b] if cmap_coeffs.ndim == 5 else cmap_coeffs,
                mask[b] if mask.ndim == 2 else mask,
                box[b] if box.ndim == 2 else box,
            )
        )(jnp.arange(B))
    return _one(positions, cmap_torsions, map_indices, cmap_coeffs, mask, box)


def make_cmap_torsions_op(
    *,
    has_box: bool = False,
    first_is_psi: bool = False,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for 5-body CMAP bicubic spline torsions.

    Signature of the returned op:
    ``(positions, cmap_torsions, map_indices, cmap_coeffs, mask, box) -> energy``.
    """
    hb = bool(has_box)
    fpsi = bool(first_is_psi)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(cmap_torsions_reference, has_box=hb, first_is_psi=fpsi)

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(pos, tors, midx, coeffs, msk, bx):
            return _pallas_cmap_torsions_fwd(
                pos, tors, midx, coeffs, msk, bx, hb, fpsi, interp, cpg
            )

        def _bwd(res, ct):
            return _pallas_cmap_torsions_bwd(hb, fpsi, interp, cpg, res, ct)

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_cmap_torsions",
        context=context,
    )


def pallas_cmap_torsions(
    positions: jax.Array,
    cmap_torsions: jax.Array | None,
    map_indices: jax.Array | None,
    mask_or_coeffs: jax.Array | None = None,
    cmap_coeffs: jax.Array | None = None,
    box: jax.Array | None = None,
    *,
    mask: jax.Array | None = None,
    first_is_psi: bool = False,
    interpret: bool | None = None,
    compute_param_grads: bool | None = None,
) -> jax.Array:
    """1D-tiled (BLOCK_C=32) 5-body CMAP 2D bicubic spline Pallas kernel."""
    if cmap_coeffs is None:
        eff_coeffs = mask_or_coeffs
        eff_mask = mask
    else:
        eff_coeffs = cmap_coeffs
        eff_mask = mask_or_coeffs if mask is None else mask

    if (
        cmap_torsions is None
        or eff_coeffs is None
        or cmap_torsions.shape[-2] == 0
        or eff_coeffs.size == 0
        or eff_coeffs.ndim < 3
    ):
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    if cmap_torsions.shape[-1] == 8:
        tors_5col = jnp.concatenate(
            [cmap_torsions[..., :, :4], cmap_torsions[..., :, 7:8]], axis=-1
        ).astype(jnp.int32)
    else:
        tors_5col = jnp.asarray(cmap_torsions[..., :, :5], dtype=jnp.int32)
    midx_arr = (
        jnp.zeros(tors_5col.shape[:-1], dtype=jnp.int32)
        if map_indices is None
        else jnp.asarray(map_indices, dtype=jnp.int32)
    )
    msk_arr = (
        jnp.ones(tors_5col.shape[:-1], dtype=positions.dtype)
        if eff_mask is None
        else jnp.asarray(eff_mask, dtype=positions.dtype)
    )
    coeffs_arr = jnp.asarray(eff_coeffs, dtype=positions.dtype)
    if coeffs_arr.shape[-1] != 16:
        if coeffs_arr.ndim == 3:
            coeffs_arr = precompute_cmap_coefficients(coeffs_arr).astype(positions.dtype)
        elif coeffs_arr.ndim == 4:
            coeffs_arr = jax.vmap(precompute_cmap_coefficients)(coeffs_arr).astype(positions.dtype)

    box_arr, has_box = _normalize_box(box, positions)

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_cmap_torsions_op(
            has_box=bool(has_box),
            first_is_psi=bool(first_is_psi),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(positions, tors_5col, midx_arr, coeffs_arr, msk_arr, box_arr)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_cmap_torsions_diff(
        positions,
        tors_5col,
        midx_arr,
        coeffs_arr,
        msk_arr,
        box_arr,
        bool(has_box),
        bool(first_is_psi),
        bool(interpret),
        cpg,
    )


# =============================================================================
# O2.2: Single-Launch Fused Bonded Megakernel (pallas_bonded_all_fused)
# =============================================================================


def _pad_1d_int(arr: jax.Array, pad_len: int, val: int = -1) -> jax.Array:
    if pad_len <= 0:
        return arr
    return jnp.pad(arr, (0, pad_len), constant_values=val)


def _pad_1d_flt(arr: jax.Array, pad_len: int, val: float = 0.0) -> jax.Array:
    if pad_len <= 0:
        return arr
    return jnp.pad(arr, (0, pad_len), constant_values=val)


def _pallas_bonded_all_fused_single(
    positions: jax.Array,
    bond_indices: jax.Array,
    bond_params: jax.Array,
    bond_mask: jax.Array,
    angle_indices: jax.Array,
    angle_params: jax.Array,
    angle_mask: jax.Array,
    dih_indices: jax.Array,
    dih_params: jax.Array,
    dih_mask: jax.Array,
    harm_imp_indices: jax.Array,
    harm_imp_params: jax.Array,
    harm_imp_mask: jax.Array,
    exc14_pairs: jax.Array,
    exc14_sigmas: jax.Array,
    exc14_epsilons: jax.Array,
    exc14_chargeprods: jax.Array,
    exc14_mask: jax.Array,
    cmap_torsions: jax.Array,
    cmap_map_indices: jax.Array,
    cmap_coeffs: jax.Array,
    cmap_mask: jax.Array,
    box: jax.Array,
    lam: jax.Array,
    has_box: bool,
    dist_eps: float,
    b0_is_ji: bool,
    angle_regularization_mode: str,
    coulomb_constant: float,
    softcore_alpha: float,
    softcore_exponent: int,
    cmap_first_is_psi: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)

    B_real = int(bond_indices.shape[0])
    A_real = int(angle_indices.shape[0])
    D_real = int(dih_indices.shape[0])
    H_real = int(harm_imp_indices.shape[0])
    E_real = int(exc14_pairs.shape[0])
    C_real = int(cmap_torsions.shape[0]) if cmap_coeffs.size > 0 else 0

    n_blk_b = (B_real + _BLOCK_B - 1) // _BLOCK_B if B_real > 0 else 0
    n_blk_a = (A_real + _BLOCK_B - 1) // _BLOCK_B if A_real > 0 else 0
    n_blk_d = (D_real + _BLOCK_B - 1) // _BLOCK_B if D_real > 0 else 0
    n_blk_h = (H_real + _BLOCK_B - 1) // _BLOCK_B if H_real > 0 else 0
    n_blk_e = (E_real + _BLOCK_B - 1) // _BLOCK_B if E_real > 0 else 0
    n_blk_c = (C_real + _BLOCK_B - 1) // _BLOCK_B if C_real > 0 else 0

    total_blocks = n_blk_b + n_blk_a + n_blk_d + n_blk_h + n_blk_e + n_blk_c
    if total_blocks == 0:
        return jnp.zeros((), dtype=orig_dtype), jnp.zeros((N, 3), dtype=orig_dtype)

    o_b0, o_b1 = 0, n_blk_b
    o_a0, o_a1 = o_b1, o_b1 + n_blk_a
    o_d0, o_d1 = o_a1, o_a1 + n_blk_d
    o_h0, o_h1 = o_d1, o_d1 + n_blk_h
    o_e0, o_e1 = o_h1, o_h1 + n_blk_e
    o_c0, o_c1 = o_e1, o_e1 + n_blk_c

    n_atom_blocks = max(1, (N + 31) // 32)
    N_pad = n_atom_blocks * 32
    pad_n = N_pad - N
    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])

    # Pad each group to at least 1 block (_BLOCK_B) so Pallas Ref shapes are non-empty
    B_pad = max(1, n_blk_b) * _BLOCK_B
    pad_b = B_pad - B_real
    b_i = _pad_1d_int(bond_indices[:, 0].astype(jnp.int32) if B_real > 0 else jnp.zeros((0,), jnp.int32), pad_b, -1)
    b_j = _pad_1d_int(bond_indices[:, 1].astype(jnp.int32) if B_real > 0 else jnp.zeros((0,), jnp.int32), pad_b, -1)
    b_r0 = _pad_1d_flt(bond_params[:, 0].astype(dtype) if B_real > 0 else jnp.zeros((0,), dtype), pad_b, 0.0)
    b_k = _pad_1d_flt(bond_params[:, 1].astype(dtype) if B_real > 0 else jnp.zeros((0,), dtype), pad_b, 0.0)
    b_msk = _pad_1d_flt(bond_mask.astype(dtype) if B_real > 0 else jnp.zeros((0,), dtype), pad_b, 0.0)

    A_pad = max(1, n_blk_a) * _BLOCK_B
    pad_a = A_pad - A_real
    a_i = _pad_1d_int(angle_indices[:, 0].astype(jnp.int32) if A_real > 0 else jnp.zeros((0,), jnp.int32), pad_a, -1)
    a_j = _pad_1d_int(angle_indices[:, 1].astype(jnp.int32) if A_real > 0 else jnp.zeros((0,), jnp.int32), pad_a, -1)
    a_k = _pad_1d_int(angle_indices[:, 2].astype(jnp.int32) if A_real > 0 else jnp.zeros((0,), jnp.int32), pad_a, -1)
    a_th0 = _pad_1d_flt(angle_params[:, 0].astype(dtype) if A_real > 0 else jnp.zeros((0,), dtype), pad_a, 0.0)
    a_kval = _pad_1d_flt(angle_params[:, 1].astype(dtype) if A_real > 0 else jnp.zeros((0,), dtype), pad_a, 0.0)
    a_msk = _pad_1d_flt(angle_mask.astype(dtype) if A_real > 0 else jnp.zeros((0,), dtype), pad_a, 0.0)

    D_pad = max(1, n_blk_d) * _BLOCK_B
    pad_d = D_pad - D_real
    d_i = _pad_1d_int(dih_indices[:, 0].astype(jnp.int32) if D_real > 0 else jnp.zeros((0,), jnp.int32), pad_d, -1)
    d_j = _pad_1d_int(dih_indices[:, 1].astype(jnp.int32) if D_real > 0 else jnp.zeros((0,), jnp.int32), pad_d, -1)
    d_k = _pad_1d_int(dih_indices[:, 2].astype(jnp.int32) if D_real > 0 else jnp.zeros((0,), jnp.int32), pad_d, -1)
    d_l = _pad_1d_int(dih_indices[:, 3].astype(jnp.int32) if D_real > 0 else jnp.zeros((0,), jnp.int32), pad_d, -1)
    d_p0 = _pad_1d_flt(dih_params[:, 0].astype(dtype) if D_real > 0 else jnp.zeros((0,), dtype), pad_d, 0.0)
    d_p1 = _pad_1d_flt(dih_params[:, 1].astype(dtype) if D_real > 0 else jnp.zeros((0,), dtype), pad_d, 0.0)
    d_p2 = _pad_1d_flt(dih_params[:, 2].astype(dtype) if (D_real > 0 and dih_params.shape[-1] >= 3) else jnp.zeros((0,), dtype), pad_d, 0.0)
    d_msk = _pad_1d_flt(dih_mask.astype(dtype) if D_real > 0 else jnp.zeros((0,), dtype), pad_d, 0.0)

    H_pad = max(1, n_blk_h) * _BLOCK_B
    pad_h = H_pad - H_real
    h_i = _pad_1d_int(harm_imp_indices[:, 0].astype(jnp.int32) if H_real > 0 else jnp.zeros((0,), jnp.int32), pad_h, -1)
    h_j = _pad_1d_int(harm_imp_indices[:, 1].astype(jnp.int32) if H_real > 0 else jnp.zeros((0,), jnp.int32), pad_h, -1)
    h_k = _pad_1d_int(harm_imp_indices[:, 2].astype(jnp.int32) if H_real > 0 else jnp.zeros((0,), jnp.int32), pad_h, -1)
    h_l = _pad_1d_int(harm_imp_indices[:, 3].astype(jnp.int32) if H_real > 0 else jnp.zeros((0,), jnp.int32), pad_h, -1)
    h_kval = _pad_1d_flt(harm_imp_params[:, 0].astype(dtype) if H_real > 0 else jnp.zeros((0,), dtype), pad_h, 0.0)
    h_phi0 = _pad_1d_flt(harm_imp_params[:, 1].astype(dtype) if H_real > 0 else jnp.zeros((0,), dtype), pad_h, 0.0)
    h_msk = _pad_1d_flt(harm_imp_mask.astype(dtype) if H_real > 0 else jnp.zeros((0,), dtype), pad_h, 0.0)

    E_pad = max(1, n_blk_e) * _BLOCK_B
    pad_e = E_pad - E_real
    e_i = _pad_1d_int(exc14_pairs[:, 0].astype(jnp.int32) if E_real > 0 else jnp.zeros((0,), jnp.int32), pad_e, -1)
    e_j = _pad_1d_int(exc14_pairs[:, 1].astype(jnp.int32) if E_real > 0 else jnp.zeros((0,), jnp.int32), pad_e, -1)
    e_sig = _pad_1d_flt(exc14_sigmas.astype(dtype) if E_real > 0 else jnp.zeros((0,), dtype), pad_e, 1.0)
    e_eps = _pad_1d_flt(exc14_epsilons.astype(dtype) if E_real > 0 else jnp.zeros((0,), dtype), pad_e, 0.0)
    e_qp = _pad_1d_flt(exc14_chargeprods.astype(dtype) if E_real > 0 else jnp.zeros((0,), dtype), pad_e, 0.0)
    e_msk = _pad_1d_flt(exc14_mask.astype(dtype) if E_real > 0 else jnp.zeros((0,), dtype), pad_e, 0.0)

    C_pad = max(1, n_blk_c) * _BLOCK_B
    pad_c = C_pad - C_real
    c_0 = _pad_1d_int(cmap_torsions[:, 0].astype(jnp.int32) if C_real > 0 else jnp.zeros((0,), jnp.int32), pad_c, -1)
    c_1 = _pad_1d_int(cmap_torsions[:, 1].astype(jnp.int32) if C_real > 0 else jnp.zeros((0,), jnp.int32), pad_c, -1)
    c_2 = _pad_1d_int(cmap_torsions[:, 2].astype(jnp.int32) if C_real > 0 else jnp.zeros((0,), jnp.int32), pad_c, -1)
    c_3 = _pad_1d_int(cmap_torsions[:, 3].astype(jnp.int32) if C_real > 0 else jnp.zeros((0,), jnp.int32), pad_c, -1)
    c_4 = _pad_1d_int(cmap_torsions[:, 4].astype(jnp.int32) if C_real > 0 else jnp.zeros((0,), jnp.int32), pad_c, -1)
    c_midx = _pad_1d_int(cmap_map_indices.astype(jnp.int32) if C_real > 0 else jnp.zeros((0,), jnp.int32), pad_c, 0)
    c_msk = _pad_1d_flt(cmap_mask.astype(dtype) if C_real > 0 else jnp.zeros((0,), dtype), pad_c, 0.0)

    if cmap_coeffs.size > 0 and cmap_coeffs.ndim >= 4:
        N_maps = int(cmap_coeffs.shape[0])
        G_cmap = int(cmap_coeffs.shape[1])
        c_coeffs_flat = jnp.asarray(cmap_coeffs.astype(dtype).reshape(-1))
    else:
        N_maps = 1
        G_cmap = 24
        c_coeffs_flat = jnp.zeros((16,), dtype=dtype)

    box_vec = jnp.asarray(box.astype(dtype))
    lam_vec = jnp.asarray(lam.reshape((1,)).astype(dtype))
    zero_f = jnp.zeros((N_pad,), dtype=dtype)

    def _bonded_all_fused_kernel(
        px_ref,
        py_ref,
        pz_ref,
        b_i_ref,
        b_j_ref,
        b_r0_ref,
        b_k_ref,
        b_msk_ref,
        a_i_ref,
        a_j_ref,
        a_k_ref,
        a_th0_ref,
        a_kval_ref,
        a_msk_ref,
        d_i_ref,
        d_j_ref,
        d_k_ref,
        d_l_ref,
        d_p0_ref,
        d_p1_ref,
        d_p2_ref,
        d_msk_ref,
        h_i_ref,
        h_j_ref,
        h_k_ref,
        h_l_ref,
        h_kval_ref,
        h_phi0_ref,
        h_msk_ref,
        e_i_ref,
        e_j_ref,
        e_sig_ref,
        e_eps_ref,
        e_qp_ref,
        e_msk_ref,
        c_0_ref,
        c_1_ref,
        c_2_ref,
        c_3_ref,
        c_4_ref,
        c_midx_ref,
        c_coeffs_flat_ref,
        c_msk_ref,
        box_ref,
        lam_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        pid = pl.program_id(0)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)
        six = jnp.asarray(6.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
        max_atom = max(N - 1, 0)

        def _do_bonds(p_val):
            offs = (p_val - o_b0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            ai = _pl_load(b_i_ref, (offs,))
            aj = _pl_load(b_j_ref, (offs,))
            mv = _pl_load(b_msk_ref, (offs,))
            valid = (offs < B_real) & (mv > zero) & (ai >= 0) & (ai < N) & (aj >= 0) & (aj < N)
            m_eff = jnp.where(valid, mv, zero)
            si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
            sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
            ri = (_pl_load(px_ref, (si,)), _pl_load(py_ref, (si,)), _pl_load(pz_ref, (si,)))
            rj = (_pl_load(px_ref, (sj,)), _pl_load(py_ref, (sj,)), _pl_load(pz_ref, (sj,)))
            d_vec = _min_image3(_sub3(ri, rj), _load_box6(box_ref), has_box)
            d_sq = _dot3(d_vec, d_vec)
            if dist_eps > 0.0:
                dist = jnp.sqrt(d_sq + jnp.asarray(dist_eps, dtype=dtype))
            else:
                d_sq_s = jnp.where(d_sq > zero, d_sq, one)
                dist = jnp.where(d_sq > zero, jnp.sqrt(d_sq_s), zero)
            r0_b = _pl_load(b_r0_ref, (offs,))
            k_b = _pl_load(b_k_ref, (offs,))
            diff = dist - r0_b
            e_b = half * m_eff * k_b * diff * diff
            dist_denom = jnp.where(dist > zero, dist, one)
            f_coeff = jnp.where(valid & (dist > zero), (zero - m_eff * k_b * diff) / dist_denom, zero)
            fi = _scale3(f_coeff, d_vec)
            fj = _scale3(zero - one, fi)
            _pl_atomic_add_1d(fx_out_ref, si, fi[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, si, fi[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, si, fi[2], valid, interpret)
            _pl_atomic_add_1d(fx_out_ref, sj, fj[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, sj, fj[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, sj, fj[2], valid, interpret)
            return jnp.sum(e_b)

        def _do_angles(p_val):
            offs = (p_val - o_a0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            ai = _pl_load(a_i_ref, (offs,))
            aj = _pl_load(a_j_ref, (offs,))
            ak = _pl_load(a_k_ref, (offs,))
            mv = _pl_load(a_msk_ref, (offs,))
            valid = (
                (offs < A_real)
                & (mv > zero)
                & (ai >= 0)
                & (ai < N)
                & (aj >= 0)
                & (aj < N)
                & (ak >= 0)
                & (ak < N)
            )
            m_eff = jnp.where(valid, mv, zero)
            si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
            sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
            sk = jnp.where(valid, jnp.clip(ak, 0, max_atom), 0)
            ri = (_pl_load(px_ref, (si,)), _pl_load(py_ref, (si,)), _pl_load(pz_ref, (si,)))
            rj = (_pl_load(px_ref, (sj,)), _pl_load(py_ref, (sj,)), _pl_load(pz_ref, (sj,)))
            rk = (_pl_load(px_ref, (sk,)), _pl_load(py_ref, (sk,)), _pl_load(pz_ref, (sk,)))
            box_xyz = _load_box6(box_ref)
            v_ji = _min_image3(_sub3(ri, rj), box_xyz, has_box)
            v_jk = _min_image3(_sub3(rk, rj), box_xyz, has_box)
            d_sq_ji = _dot3(v_ji, v_ji)
            d_sq_jk = _dot3(v_jk, v_jk)
            if dist_eps > 0.0:
                eps_val = jnp.asarray(dist_eps, dtype=dtype)
                d_ji_sq_eff = d_sq_ji + eps_val
                d_jk_sq_eff = d_sq_jk + eps_val
                d_ji = jnp.sqrt(d_ji_sq_eff)
                d_jk = jnp.sqrt(d_jk_sq_eff)
            else:
                d_ji_sq_eff = jnp.where(d_sq_ji > zero, d_sq_ji, one)
                d_jk_sq_eff = jnp.where(d_sq_jk > zero, d_sq_jk, one)
                d_ji = jnp.where(d_sq_ji > zero, jnp.sqrt(d_ji_sq_eff), zero)
                d_jk = jnp.where(d_sq_jk > zero, jnp.sqrt(d_jk_sq_eff), zero)
            dot_ik = _dot3(v_ji, v_jk)
            cp = _cross3(v_ji, v_jk)
            th0 = _pl_load(a_th0_ref, (offs,))
            k_a = _pl_load(a_kval_ref, (offs,))
            if angle_regularization_mode == "openmm_cross":
                cp_sq = _dot3(cp, cp)
                r_cp = jnp.sqrt(jnp.maximum(cp_sq, jnp.asarray(1e-24, dtype=dtype)))
                theta = jnp.arctan2(r_cp, dot_ik)
                dtheta = theta - th0
                e_a = half * m_eff * k_a * dtheta * dtheta
                neg_de_dth = jnp.where(
                    valid & (cp_sq > jnp.asarray(1e-24, dtype=dtype)) & (d_ji > zero) & (d_jk > zero),
                    zero - m_eff * k_a * dtheta,
                    zero,
                )
                fi = _scale3(neg_de_dth / (d_ji_sq_eff * r_cp), _cross3(v_ji, cp))
                fk = _scale3(neg_de_dth / (d_jk_sq_eff * r_cp), _cross3(cp, v_jk))
                fj = _scale3(zero - one, _add3(fi, fk))
            else:
                denom = d_ji * d_jk + jnp.asarray(1e-8, dtype=dtype)
                c_raw = dot_ik / denom
                c_lim = jnp.asarray(0.999999, dtype=dtype)
                c = jnp.clip(c_raw, zero - c_lim, c_lim)
                theta = jnp.arccos(c)
                dtheta = theta - th0
                e_a = half * m_eff * k_a * dtheta * dtheta
                in_clip = (c_raw > (zero - c_lim)) & (c_raw < c_lim)
                sin_th = jnp.sqrt(jnp.maximum(one - c * c, jnp.asarray(1e-12, dtype=dtype)))
                neg_de_dth = jnp.where(
                    valid & in_clip & (d_ji > zero) & (d_jk > zero),
                    zero - m_eff * k_a * dtheta,
                    zero,
                )
                r_cp_reg = denom * sin_th
                fi = _scale3(neg_de_dth / (d_ji_sq_eff * r_cp_reg), _cross3(v_ji, cp))
                fk = _scale3(neg_de_dth / (d_jk_sq_eff * r_cp_reg), _cross3(cp, v_jk))
                fj = _scale3(zero - one, _add3(fi, fk))
            _pl_atomic_add_1d(fx_out_ref, si, fi[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, si, fi[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, si, fi[2], valid, interpret)
            _pl_atomic_add_1d(fx_out_ref, sj, fj[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, sj, fj[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, sj, fj[2], valid, interpret)
            _pl_atomic_add_1d(fx_out_ref, sk, fk[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, sk, fk[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, sk, fk[2], valid, interpret)
            return jnp.sum(e_a)

        def _do_torsions_periodic(p_val):
            offs = (p_val - o_d0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            ai = _pl_load(d_i_ref, (offs,))
            aj = _pl_load(d_j_ref, (offs,))
            ak = _pl_load(d_k_ref, (offs,))
            al = _pl_load(d_l_ref, (offs,))
            mv = _pl_load(d_msk_ref, (offs,))
            valid = (
                (offs < D_real)
                & (mv > zero)
                & (ai >= 0)
                & (ai < N)
                & (aj >= 0)
                & (aj < N)
                & (ak >= 0)
                & (ak < N)
                & (al >= 0)
                & (al < N)
            )
            m_eff = jnp.where(valid, mv, zero)
            si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
            sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
            sk = jnp.where(valid, jnp.clip(ak, 0, max_atom), 0)
            sl = jnp.where(valid, jnp.clip(al, 0, max_atom), 0)
            ri = (_pl_load(px_ref, (si,)), _pl_load(py_ref, (si,)), _pl_load(pz_ref, (si,)))
            rj = (_pl_load(px_ref, (sj,)), _pl_load(py_ref, (sj,)), _pl_load(pz_ref, (sj,)))
            rk = (_pl_load(px_ref, (sk,)), _pl_load(py_ref, (sk,)), _pl_load(pz_ref, (sk,)))
            rl = (_pl_load(px_ref, (sl,)), _pl_load(py_ref, (sl,)), _pl_load(pz_ref, (sl,)))
            phi, dphi_dri, dphi_drj, dphi_drk, dphi_drl = _torsion_angle_and_atom_grads_vec3(
                ri, rj, rk, rl, valid, _load_box6(box_ref), has_box, b0_is_ji
            )
            prm0 = _pl_load(d_p0_ref, (offs,))
            prm1 = _pl_load(d_p1_ref, (offs,))
            prm2 = _pl_load(d_p2_ref, (offs,))
            arg = prm0 * phi - prm1
            e_d = m_eff * prm2 * (one + jnp.cos(arg))
            g_phi = (zero - m_eff) * prm2 * prm0 * jnp.sin(arg)
            neg_g_phi = jnp.where(valid, zero - g_phi, zero)
            fi = _scale3(neg_g_phi, dphi_dri)
            fj = _scale3(neg_g_phi, dphi_drj)
            fk = _scale3(neg_g_phi, dphi_drk)
            fl = _scale3(neg_g_phi, dphi_drl)
            for s_atom, f_atom in ((si, fi), (sj, fj), (sk, fk), (sl, fl)):
                _pl_atomic_add_1d(fx_out_ref, s_atom, f_atom[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, s_atom, f_atom[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, s_atom, f_atom[2], valid, interpret)
            return jnp.sum(e_d)

        def _do_impropers_harmonic(p_val):
            offs = (p_val - o_h0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            ai = _pl_load(h_i_ref, (offs,))
            aj = _pl_load(h_j_ref, (offs,))
            ak = _pl_load(h_k_ref, (offs,))
            al = _pl_load(h_l_ref, (offs,))
            mv = _pl_load(h_msk_ref, (offs,))
            valid = (
                (offs < H_real)
                & (mv > zero)
                & (ai >= 0)
                & (ai < N)
                & (aj >= 0)
                & (aj < N)
                & (ak >= 0)
                & (ak < N)
                & (al >= 0)
                & (al < N)
            )
            m_eff = jnp.where(valid, mv, zero)
            si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
            sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
            sk = jnp.where(valid, jnp.clip(ak, 0, max_atom), 0)
            sl = jnp.where(valid, jnp.clip(al, 0, max_atom), 0)
            ri = (_pl_load(px_ref, (si,)), _pl_load(py_ref, (si,)), _pl_load(pz_ref, (si,)))
            rj = (_pl_load(px_ref, (sj,)), _pl_load(py_ref, (sj,)), _pl_load(pz_ref, (sj,)))
            rk = (_pl_load(px_ref, (sk,)), _pl_load(py_ref, (sk,)), _pl_load(pz_ref, (sk,)))
            rl = (_pl_load(px_ref, (sl,)), _pl_load(py_ref, (sl,)), _pl_load(pz_ref, (sl,)))
            phi, dphi_dri, dphi_drj, dphi_drk, dphi_drl = _torsion_angle_and_atom_grads_vec3(
                ri, rj, rk, rl, valid, _load_box6(box_ref), has_box, b0_is_ji
            )
            k_h = _pl_load(h_kval_ref, (offs,))
            phi0_h = _pl_load(h_phi0_ref, (offs,))
            diff_raw = phi - phi0_h + pi
            dphi = diff_raw - two_pi * jnp.floor(diff_raw / two_pi) - pi
            e_h = m_eff * k_h * dphi * dphi
            g_phi = two * m_eff * k_h * dphi
            neg_g_phi = jnp.where(valid, zero - g_phi, zero)
            fi = _scale3(neg_g_phi, dphi_dri)
            fj = _scale3(neg_g_phi, dphi_drj)
            fk = _scale3(neg_g_phi, dphi_drk)
            fl = _scale3(neg_g_phi, dphi_drl)
            for s_atom, f_atom in ((si, fi), (sj, fj), (sk, fk), (sl, fl)):
                _pl_atomic_add_1d(fx_out_ref, s_atom, f_atom[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, s_atom, f_atom[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, s_atom, f_atom[2], valid, interpret)
            return jnp.sum(e_h)

        def _do_exceptions_14(p_val):
            offs = (p_val - o_e0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            coul = jnp.asarray(coulomb_constant, dtype=dtype)
            ai = _pl_load(e_i_ref, (offs,))
            aj = _pl_load(e_j_ref, (offs,))
            mv = _pl_load(e_msk_ref, (offs,))
            valid = (offs < E_real) & (mv > zero) & (ai >= 0) & (ai < N) & (aj >= 0) & (aj < N) & (ai != aj)
            m_eff = jnp.where(valid, mv, zero)
            si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
            sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
            ri = (_pl_load(px_ref, (si,)), _pl_load(py_ref, (si,)), _pl_load(pz_ref, (si,)))
            rj = (_pl_load(px_ref, (sj,)), _pl_load(py_ref, (sj,)), _pl_load(pz_ref, (sj,)))
            lam_v = _pl_load(lam_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
            d_vec = _min_image3(_sub3(ri, rj), _load_box6(box_ref), has_box)
            r_sq = _dot3(d_vec, d_vec) + jnp.asarray(1e-12, dtype=dtype)
            dist = jnp.sqrt(r_sq)
            sig_e = _pl_load(e_sig_ref, (offs,))
            eps_e = _pl_load(e_eps_ref, (offs,))
            qp_e = _pl_load(e_qp_ref, (offs,))
            sig_safe = jnp.where(valid, jnp.maximum(sig_e, jnp.asarray(1e-4, dtype=dtype)), one)
            one_minus_lam = jnp.maximum(one - lam_v, zero)
            soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * (one_minus_lam ** int(softcore_exponent))
            r6 = (dist / sig_safe) ** 6
            denom = soft_term + r6 + jnp.asarray(1e-12, dtype=dtype)
            inv_d = one / denom
            inv_d2 = inv_d * inv_d
            inv_d3 = inv_d2 * inv_d
            e_lj = four * eps_e * lam_v * (inv_d2 - inv_d)
            e_coul = coul * qp_e / dist
            e_pair = m_eff * (e_lj + e_coul)
            de_dr_over_r = (m_eff / r_sq) * (
                four * eps_e * lam_v * (inv_d2 - two * inv_d3) * (six * r6) - e_coul
            )
            f_coeff = jnp.where(valid, zero - de_dr_over_r, zero)
            fi = _scale3(f_coeff, d_vec)
            fj = _scale3(zero - one, fi)
            _pl_atomic_add_1d(fx_out_ref, si, fi[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, si, fi[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, si, fi[2], valid, interpret)
            _pl_atomic_add_1d(fx_out_ref, sj, fj[0], valid, interpret)
            _pl_atomic_add_1d(fy_out_ref, sj, fj[1], valid, interpret)
            _pl_atomic_add_1d(fz_out_ref, sj, fj[2], valid, interpret)
            return jnp.sum(e_pair)

        def _do_cmap(p_val):
            offs = (p_val - o_c0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            delta = jnp.asarray(2.0 * jnp.pi / G_cmap, dtype=dtype)
            inv_delta = one / delta
            i0 = _pl_load(c_0_ref, (offs,))
            i1 = _pl_load(c_1_ref, (offs,))
            i2 = _pl_load(c_2_ref, (offs,))
            i3 = _pl_load(c_3_ref, (offs,))
            i4 = _pl_load(c_4_ref, (offs,))
            mv = _pl_load(c_msk_ref, (offs,))
            valid = (
                (offs < C_real)
                & (mv > zero)
                & (i0 >= 0)
                & (i0 < N)
                & (i1 >= 0)
                & (i1 < N)
                & (i2 >= 0)
                & (i2 < N)
                & (i3 >= 0)
                & (i3 < N)
                & (i4 >= 0)
                & (i4 < N)
            )
            m_eff = jnp.where(valid, mv, zero)
            s0 = jnp.where(valid, jnp.clip(i0, 0, max_atom), 0)
            s1 = jnp.where(valid, jnp.clip(i1, 0, max_atom), 0)
            s2 = jnp.where(valid, jnp.clip(i2, 0, max_atom), 0)
            s3 = jnp.where(valid, jnp.clip(i3, 0, max_atom), 0)
            s4 = jnp.where(valid, jnp.clip(i4, 0, max_atom), 0)
            r0 = (_pl_load(px_ref, (s0,)), _pl_load(py_ref, (s0,)), _pl_load(pz_ref, (s0,)))
            r1 = (_pl_load(px_ref, (s1,)), _pl_load(py_ref, (s1,)), _pl_load(pz_ref, (s1,)))
            r2 = (_pl_load(px_ref, (s2,)), _pl_load(py_ref, (s2,)), _pl_load(pz_ref, (s2,)))
            r3 = (_pl_load(px_ref, (s3,)), _pl_load(py_ref, (s3,)), _pl_load(pz_ref, (s3,)))
            r4 = (_pl_load(px_ref, (s4,)), _pl_load(py_ref, (s4,)), _pl_load(pz_ref, (s4,)))
            box_xyz = _load_box6(box_ref)
            phi, dphi_0, dphi_1, dphi_2, dphi_3 = _torsion_angle_and_atom_grads_vec3(
                r0, r1, r2, r3, valid, box_xyz, has_box, False
            )
            psi, dpsi_1, dpsi_2, dpsi_3, dpsi_4 = _torsion_angle_and_atom_grads_vec3(
                r1, r2, r3, r4, valid, box_xyz, has_box, False
            )
            theta1, theta2 = (psi, phi) if cmap_first_is_psi else (phi, psi)
            th1_shift = theta1 + two_pi
            th1_norm = th1_shift - two_pi * jnp.floor(th1_shift / two_pi)
            th2_shift = theta2 + two_pi
            th2_norm = th2_shift - two_pi * jnp.floor(th2_shift / two_pi)
            q1 = th1_norm * inv_delta
            q2 = th2_norm * inv_delta
            g1_raw = jnp.floor(q1).astype(jnp.int32)
            g2_raw = jnp.floor(q2).astype(jnp.int32)
            g1 = g1_raw % G_cmap
            g2 = g2_raw % G_cmap
            u = q1 - g1_raw.astype(dtype)
            v = q2 - g2_raw.astype(dtype)
            m_raw = _pl_load(c_midx_ref, (offs,))
            m_safe = jnp.where(valid, jnp.clip(m_raw, 0, max(N_maps - 1, 0)), 0)
            base_idx = ((m_safe * G_cmap + g1) * G_cmap + g2) * 16
            e_poly, de_du, de_dv = _eval_cmap_horner_cell(
                c_coeffs_flat_ref, base_idx, u, v, dtype
            )
            e_c = m_eff * e_poly
            de_dtheta1 = m_eff * de_du * inv_delta
            de_dtheta2 = m_eff * de_dv * inv_delta
            if cmap_first_is_psi:
                de_dpsi, de_dphi = de_dtheta1, de_dtheta2
            else:
                de_dphi, de_dpsi = de_dtheta1, de_dtheta2
            neg_dphi = jnp.where(valid, zero - de_dphi, zero)
            neg_dpsi = jnp.where(valid, zero - de_dpsi, zero)
            f0 = _scale3(neg_dphi, dphi_0)
            f1 = _add3(_scale3(neg_dphi, dphi_1), _scale3(neg_dpsi, dpsi_1))
            f2 = _add3(_scale3(neg_dphi, dphi_2), _scale3(neg_dpsi, dpsi_2))
            f3 = _add3(_scale3(neg_dphi, dphi_3), _scale3(neg_dpsi, dpsi_3))
            f4 = _scale3(neg_dpsi, dpsi_4)
            for s_atom, f_atom in ((s0, f0), (s1, f1), (s2, f2), (s3, f3), (s4, f4)):
                _pl_atomic_add_1d(fx_out_ref, s_atom, f_atom[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, s_atom, f_atom[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, s_atom, f_atom[2], valid, interpret)
            return jnp.sum(e_c)

        active_groups = []
        if n_blk_b > 0:
            active_groups.append((o_b0, o_b1, _do_bonds))
        if n_blk_a > 0:
            active_groups.append((o_a0, o_a1, _do_angles))
        if n_blk_d > 0:
            active_groups.append((o_d0, o_d1, _do_torsions_periodic))
        if n_blk_h > 0:
            active_groups.append((o_h0, o_h1, _do_impropers_harmonic))
        if n_blk_e > 0:
            active_groups.append((o_e0, o_e1, _do_exceptions_14))
        if n_blk_c > 0:
            active_groups.append((o_c0, o_c1, _do_cmap))

        def _dispatch_group(g_idx: int, p_val: jax.Array) -> jax.Array:
            if g_idx == len(active_groups) - 1:
                return active_groups[g_idx][2](p_val)
            _, o_end, fn_g = active_groups[g_idx]
            return jax.lax.cond(
                p_val < o_end,
                fn_g,
                lambda p: _dispatch_group(g_idx + 1, p),
                p_val,
            )

        e_sum = _dispatch_group(0, pid)
        _pl_store(e_out_ref, (jnp.full((1,), pid, dtype=jnp.int32),), e_sum[None])

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(48)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((total_blocks,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
        jax.ShapeDtypeStruct((N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _bonded_all_fused_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(total_blocks,),
        input_output_aliases={45: 1, 46: 2, 47: 3},
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_bonded_all_fused",
    )(
        px,
        py,
        pz,
        b_i,
        b_j,
        b_r0,
        b_k,
        b_msk,
        a_i,
        a_j,
        a_k,
        a_th0,
        a_kval,
        a_msk,
        d_i,
        d_j,
        d_k,
        d_l,
        d_p0,
        d_p1,
        d_p2,
        d_msk,
        h_i,
        h_j,
        h_k,
        h_l,
        h_kval,
        h_phi0,
        h_msk,
        e_i,
        e_j,
        e_sig,
        e_eps,
        e_qp,
        e_msk,
        c_0,
        c_1,
        c_2,
        c_3,
        c_4,
        c_midx,
        c_coeffs_flat,
        c_msk,
        box_vec,
        lam_vec,
        zero_f,
        zero_f,
        zero_f,
    )
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_blk).astype(orig_dtype), forces


def _bonded_all_fused_param_grads(
    positions: jax.Array,
    bond_indices: jax.Array,
    bond_params: jax.Array,
    bond_mask: jax.Array,
    angle_indices: jax.Array,
    angle_params: jax.Array,
    angle_mask: jax.Array,
    dih_indices: jax.Array,
    dih_params: jax.Array,
    dih_mask: jax.Array,
    harm_imp_indices: jax.Array,
    harm_imp_params: jax.Array,
    harm_imp_mask: jax.Array,
    exc14_pairs: jax.Array,
    exc14_sigmas: jax.Array,
    exc14_epsilons: jax.Array,
    exc14_chargeprods: jax.Array,
    exc14_mask: jax.Array,
    cmap_torsions: jax.Array,
    cmap_map_indices: jax.Array,
    cmap_coeffs: jax.Array,
    cmap_mask: jax.Array,
    box: jax.Array,
    lam: jax.Array,
    has_box: bool,
    dist_eps: float,
    b0_is_ji: bool,
    coulomb_constant: float,
    softcore_alpha: float,
    softcore_exponent: int,
    cmap_first_is_psi: bool,
):
    g_bond = (
        _bonds_2body_param_grads(positions, bond_indices, bond_params, bond_mask, box, has_box, dist_eps)
        if bond_indices.shape[0] > 0
        else jnp.zeros_like(bond_params)
    )
    g_angle = (
        _angles_harmonic_param_grads(positions, angle_indices, angle_params, angle_mask, box, has_box, dist_eps)
        if angle_indices.shape[0] > 0
        else jnp.zeros_like(angle_params)
    )
    g_dih = (
        _torsions_4body_param_grads(positions, dih_indices, dih_params, dih_mask, box, has_box, b0_is_ji, False)
        if dih_indices.shape[0] > 0
        else jnp.zeros_like(dih_params)
    )
    g_himp = (
        _torsions_4body_param_grads(positions, harm_imp_indices, harm_imp_params, harm_imp_mask, box, has_box, b0_is_ji, True)
        if harm_imp_indices.shape[0] > 0
        else jnp.zeros_like(harm_imp_params)
    )
    if exc14_pairs.shape[0] > 0:
        g_esig, g_eeps, g_eqp, g_lam = _exceptions_14_param_grads(
            positions,
            exc14_pairs,
            exc14_sigmas,
            exc14_epsilons,
            exc14_chargeprods,
            exc14_mask,
            box,
            lam,
            has_box,
            coulomb_constant,
            softcore_alpha,
            softcore_exponent,
        )
    else:
        g_esig = jnp.zeros_like(exc14_sigmas)
        g_eeps = jnp.zeros_like(exc14_epsilons)
        g_eqp = jnp.zeros_like(exc14_chargeprods)
        g_lam = jnp.zeros_like(lam)
    g_cmap = (
        _cmap_coeffs_grad(
            positions,
            cmap_torsions,
            cmap_map_indices,
            cmap_coeffs,
            cmap_mask,
            box,
            has_box,
            cmap_first_is_psi,
        )
        if (cmap_torsions.shape[0] > 0 and cmap_coeffs.size > 0)
        else jnp.zeros_like(cmap_coeffs)
    )
    return g_bond, g_angle, g_dih, g_himp, g_esig, g_eeps, g_eqp, g_cmap, g_lam


def _pallas_bonded_all_fused_vmappable(
    positions,
    bond_indices,
    bond_params,
    bond_mask,
    angle_indices,
    angle_params,
    angle_mask,
    dih_indices,
    dih_params,
    dih_mask,
    harm_imp_indices,
    harm_imp_params,
    harm_imp_mask,
    exc14_pairs,
    exc14_sigmas,
    exc14_epsilons,
    exc14_chargeprods,
    exc14_mask,
    cmap_torsions,
    cmap_map_indices,
    cmap_coeffs,
    cmap_mask,
    box,
    lam,
    has_box,
    dist_eps,
    b0_is_ji,
    angle_regularization_mode,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    cmap_first_is_psi,
    interpret,
):
    hb = bool(has_box)
    de = float(dist_eps)
    b0_ji = bool(b0_is_ji)
    arm = str(angle_regularization_mode)
    cc = float(coulomb_constant)
    sa = float(softcore_alpha)
    se = int(softcore_exponent)
    fpsi = bool(cmap_first_is_psi)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(
        pos, bi, bp, bm, ai, ap, am, di, dp, dm, hi, hp, hm, ep, es, ee, eq, em, ct, cm_i, cc_c, cm_m, bx, lm
    ):
        return _pallas_bonded_all_fused_single(
            pos, bi, bp, bm, ai, ap, am, di, dp, dm, hi, hp, hm, ep, es, ee, eq, em, ct, cm_i, cc_c, cm_m, bx, lm,
            hb, de, b0_ji, arm, cc, sa, se, fpsi, interp,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, *args):
        def _step(b_idx):
            step_args = [a[b_idx] if ib else a for a, ib in zip(args, in_batched)]
            return _vmappable(*step_args)

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    all_args = (
        positions,
        bond_indices,
        bond_params,
        bond_mask,
        angle_indices,
        angle_params,
        angle_mask,
        dih_indices,
        dih_params,
        dih_mask,
        harm_imp_indices,
        harm_imp_params,
        harm_imp_mask,
        exc14_pairs,
        exc14_sigmas,
        exc14_epsilons,
        exc14_chargeprods,
        exc14_mask,
        cmap_torsions,
        cmap_map_indices,
        cmap_coeffs,
        cmap_mask,
        box,
        lam,
    )
    if positions.ndim == 3:
        B = positions.shape[0]
        base_ndims = (2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 4, 1, 1, 0)
        return jax.vmap(
            lambda b: _vmappable(
                *[a[b] if a.ndim > bn else a for a, bn in zip(all_args, base_ndims)]
            )
        )(jnp.arange(B))
    return _vmappable(*all_args)


@functools.partial(custom_vjp, nondiff_argnums=(24, 25, 26, 27, 28, 29, 30, 31, 32, 33))
def _pallas_bonded_all_fused_diff(
    positions,
    bond_indices,
    bond_params,
    bond_mask,
    angle_indices,
    angle_params,
    angle_mask,
    dih_indices,
    dih_params,
    dih_mask,
    harm_imp_indices,
    harm_imp_params,
    harm_imp_mask,
    exc14_pairs,
    exc14_sigmas,
    exc14_epsilons,
    exc14_chargeprods,
    exc14_mask,
    cmap_torsions,
    cmap_map_indices,
    cmap_coeffs,
    cmap_mask,
    box,
    lam,
    has_box,
    dist_eps,
    b0_is_ji,
    angle_regularization_mode,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    cmap_first_is_psi,
    interpret,
    compute_param_grads=None,
):
    energy, _ = _pallas_bonded_all_fused_vmappable(
        positions,
        bond_indices,
        bond_params,
        bond_mask,
        angle_indices,
        angle_params,
        angle_mask,
        dih_indices,
        dih_params,
        dih_mask,
        harm_imp_indices,
        harm_imp_params,
        harm_imp_mask,
        exc14_pairs,
        exc14_sigmas,
        exc14_epsilons,
        exc14_chargeprods,
        exc14_mask,
        cmap_torsions,
        cmap_map_indices,
        cmap_coeffs,
        cmap_mask,
        box,
        lam,
        has_box,
        dist_eps,
        b0_is_ji,
        angle_regularization_mode,
        coulomb_constant,
        softcore_alpha,
        softcore_exponent,
        cmap_first_is_psi,
        interpret,
    )
    return energy


def _pallas_bonded_all_fused_fwd(
    positions,
    bond_indices,
    bond_params,
    bond_mask,
    angle_indices,
    angle_params,
    angle_mask,
    dih_indices,
    dih_params,
    dih_mask,
    harm_imp_indices,
    harm_imp_params,
    harm_imp_mask,
    exc14_pairs,
    exc14_sigmas,
    exc14_epsilons,
    exc14_chargeprods,
    exc14_mask,
    cmap_torsions,
    cmap_map_indices,
    cmap_coeffs,
    cmap_mask,
    box,
    lam,
    has_box,
    dist_eps,
    b0_is_ji,
    angle_regularization_mode,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    cmap_first_is_psi,
    interpret,
    compute_param_grads=None,
):
    raw_args = (
        positions,
        bond_indices,
        bond_params,
        bond_mask,
        angle_indices,
        angle_params,
        angle_mask,
        dih_indices,
        dih_params,
        dih_mask,
        harm_imp_indices,
        harm_imp_params,
        harm_imp_mask,
        exc14_pairs,
        exc14_sigmas,
        exc14_epsilons,
        exc14_chargeprods,
        exc14_mask,
        cmap_torsions,
        cmap_map_indices,
        cmap_coeffs,
        cmap_mask,
        box,
        lam,
    )
    vals = tuple(_unwrap_primal(a) for a in raw_args)
    energy, forces = _pallas_bonded_all_fused_vmappable(
        *vals,
        has_box,
        dist_eps,
        b0_is_ji,
        angle_regularization_mode,
        coulomb_constant,
        softcore_alpha,
        softcore_exponent,
        cmap_first_is_psi,
        interpret,
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    if compute_param_grads is not None:
        any_param_perturbed = bool(compute_param_grads)
    else:
        any_param_perturbed = (
            _is_perturbed(bond_params, default_pert)
            or _is_perturbed(angle_params, default_pert)
            or _is_perturbed(dih_params, default_pert)
            or _is_perturbed(harm_imp_params, default_pert)
            or _is_perturbed(exc14_sigmas, default_pert)
            or _is_perturbed(exc14_epsilons, default_pert)
            or _is_perturbed(exc14_chargeprods, default_pert)
            or _is_perturbed(cmap_coeffs, default_pert)
            or _is_perturbed(lam, default_pert)
        )
    shapes_meta = _make_shapes_meta(*vals)
    param_res = vals if any_param_perturbed else None
    return energy, (forces, shapes_meta, param_res)


def _pallas_bonded_all_fused_bwd(
    has_box,
    dist_eps,
    b0_is_ji,
    angle_regularization_mode,
    coulomb_constant,
    softcore_alpha,
    softcore_exponent,
    cmap_first_is_psi,
    interpret,
    compute_param_grads,
    res,
    g,
):
    del angle_regularization_mode, interpret, compute_param_grads
    forces, shapes_meta, param_res = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return tuple(jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta)

    grad_pos = (-g[..., None, None] * forces if g.ndim > 0 else -g * forces).astype(
        shapes_meta[0].dtype
    )
    if param_res is None:
        return (grad_pos,) + tuple(
            jnp.zeros(m.shape, dtype=m.dtype) for m in shapes_meta[1:]
        )

    (
        positions,
        bond_indices,
        bond_params,
        bond_mask,
        angle_indices,
        angle_params,
        angle_mask,
        dih_indices,
        dih_params,
        dih_mask,
        harm_imp_indices,
        harm_imp_params,
        harm_imp_mask,
        exc14_pairs,
        exc14_sigmas,
        exc14_epsilons,
        exc14_chargeprods,
        exc14_mask,
        cmap_torsions,
        cmap_map_indices,
        cmap_coeffs,
        cmap_mask,
        box,
        lam,
    ) = param_res

    if positions.ndim == 2:
        gb, ga, gd, gh, ges, gee, geq, gc, glam = _bonded_all_fused_param_grads(
            positions,
            bond_indices,
            bond_params,
            bond_mask,
            angle_indices,
            angle_params,
            angle_mask,
            dih_indices,
            dih_params,
            dih_mask,
            harm_imp_indices,
            harm_imp_params,
            harm_imp_mask,
            exc14_pairs,
            exc14_sigmas,
            exc14_epsilons,
            exc14_chargeprods,
            exc14_mask,
            cmap_torsions,
            cmap_map_indices,
            cmap_coeffs,
            cmap_mask,
            box,
            lam,
            has_box,
            dist_eps,
            b0_is_ji,
            coulomb_constant,
            softcore_alpha,
            softcore_exponent,
            cmap_first_is_psi,
        )
        grad_bp = (g * gb).astype(bond_params.dtype)
        grad_ap = (g * ga).astype(angle_params.dtype)
        grad_dp = (g * gd).astype(dih_params.dtype)
        grad_hp = (g * gh).astype(harm_imp_params.dtype)
        grad_es = (g * ges).astype(exc14_sigmas.dtype)
        grad_ee = (g * gee).astype(exc14_epsilons.dtype)
        grad_eq = (g * geq).astype(exc14_chargeprods.dtype)
        grad_cc = (g * gc).astype(cmap_coeffs.dtype)
        grad_lm = (g * glam).astype(lam.dtype)
    else:
        B = positions.shape[0]
        base_ndims = (2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 4, 1, 1, 0)

        def _step(b):
            step_in = [
                a[b] if a.ndim > bn else a
                for a, bn in zip(param_res, base_ndims)
            ]
            gb_b, ga_b, gd_b, gh_b, ges_b, gee_b, geq_b, gc_b, glam_b = _bonded_all_fused_param_grads(
                *step_in,
                has_box,
                dist_eps,
                b0_is_ji,
                coulomb_constant,
                softcore_alpha,
                softcore_exponent,
                cmap_first_is_psi,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (
                (g_b * gb_b).astype(bond_params.dtype),
                (g_b * ga_b).astype(angle_params.dtype),
                (g_b * gd_b).astype(dih_params.dtype),
                (g_b * gh_b).astype(harm_imp_params.dtype),
                (g_b * ges_b).astype(exc14_sigmas.dtype),
                (g_b * gee_b).astype(exc14_epsilons.dtype),
                (g_b * geq_b).astype(exc14_chargeprods.dtype),
                (g_b * gc_b).astype(cmap_coeffs.dtype),
                (g_b * glam_b).astype(lam.dtype),
            )

        gb_all, ga_all, gd_all, gh_all, ges_all, gee_all, geq_all, gc_all, glam_all = jax.lax.map(
            _step, jnp.arange(B)
        )
        grad_bp = gb_all if bond_params.ndim > 2 else jnp.sum(gb_all, axis=0)
        grad_ap = ga_all if angle_params.ndim > 2 else jnp.sum(ga_all, axis=0)
        grad_dp = gd_all if dih_params.ndim > 2 else jnp.sum(gd_all, axis=0)
        grad_hp = gh_all if harm_imp_params.ndim > 2 else jnp.sum(gh_all, axis=0)
        grad_es = ges_all if exc14_sigmas.ndim > 1 else jnp.sum(ges_all, axis=0)
        grad_ee = gee_all if exc14_epsilons.ndim > 1 else jnp.sum(gee_all, axis=0)
        grad_eq = geq_all if exc14_chargeprods.ndim > 1 else jnp.sum(geq_all, axis=0)
        grad_cc = gc_all if cmap_coeffs.ndim > 4 else jnp.sum(gc_all, axis=0)
        grad_lm = glam_all if lam.ndim > 0 else jnp.sum(glam_all, axis=0)

    z = lambda idx: jnp.zeros(shapes_meta[idx].shape, dtype=shapes_meta[idx].dtype)
    return (
        grad_pos,
        z(1),
        grad_bp,
        z(3),
        z(4),
        grad_ap,
        z(6),
        z(7),
        grad_dp,
        z(9),
        z(10),
        grad_hp,
        z(12),
        z(13),
        grad_es,
        grad_ee,
        grad_eq,
        z(17),
        z(18),
        z(19),
        grad_cc,
        z(21),
        z(22),
        grad_lm,
    )


_pallas_bonded_all_fused_diff.defvjp(
    _pallas_bonded_all_fused_fwd, _pallas_bonded_all_fused_bwd, symbolic_zeros=True
)


def bonded_all_fused_reference(
    positions: jax.Array,
    bond_indices: jax.Array,
    bond_params: jax.Array,
    bond_mask: jax.Array,
    angle_indices: jax.Array,
    angle_params: jax.Array,
    angle_mask: jax.Array,
    dih_indices: jax.Array,
    dih_params: jax.Array,
    dih_mask: jax.Array,
    harm_imp_indices: jax.Array,
    harm_imp_params: jax.Array,
    harm_imp_mask: jax.Array,
    exc14_pairs: jax.Array,
    exc14_sigmas: jax.Array,
    exc14_epsilons: jax.Array,
    exc14_chargeprods: jax.Array,
    exc14_mask: jax.Array,
    cmap_torsions: jax.Array,
    cmap_map_indices: jax.Array,
    cmap_coeffs: jax.Array,
    cmap_mask: jax.Array,
    box: jax.Array,
    lam: jax.Array,
    *,
    has_box: bool = False,
    dist_eps: float = 1e-10,
    b0_is_ji: bool = True,
    coulomb_constant: float = 332.0637,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    cmap_first_is_psi: bool = False,
) -> jax.Array:
    """Plain-JAX reference matching ``pallas_bonded_all_fused`` across all 6 bonded groups."""
    dtype = positions.dtype
    e_total = jnp.zeros(positions.shape[:-2], dtype=dtype)
    if bond_indices.shape[-2] > 0:
        e_total = e_total + bonds_2body_reference(
            positions, bond_indices, bond_params, bond_mask, box, has_box=has_box, dist_eps=dist_eps
        )
    if angle_indices.shape[-2] > 0:
        e_total = e_total + angles_harmonic_reference(
            positions, angle_indices, angle_params, angle_mask, box, has_box=has_box, dist_eps=dist_eps
        )
    if dih_indices.shape[-2] > 0:
        e_total = e_total + torsions_4body_reference(
            positions, dih_indices, dih_params, dih_mask, box, has_box=has_box, b0_is_ji=b0_is_ji, is_harmonic=False
        )
    if harm_imp_indices.shape[-2] > 0:
        e_total = e_total + torsions_4body_reference(
            positions, harm_imp_indices, harm_imp_params, harm_imp_mask, box, has_box=has_box, b0_is_ji=b0_is_ji, is_harmonic=True
        )
    if exc14_pairs.shape[-2] > 0:
        e_total = e_total + exceptions_14_reference(
            positions,
            exc14_pairs,
            exc14_sigmas,
            exc14_epsilons,
            exc14_chargeprods,
            exc14_mask,
            box,
            lam,
            has_box=has_box,
            coulomb_constant=coulomb_constant,
            softcore_alpha=softcore_alpha,
            softcore_exponent=softcore_exponent,
        )
    if cmap_torsions.shape[-2] > 0 and cmap_coeffs.size > 0:
        e_total = e_total + cmap_torsions_reference(
            positions,
            cmap_torsions,
            cmap_map_indices,
            cmap_coeffs,
            cmap_mask,
            box,
            has_box=has_box,
            first_is_psi=cmap_first_is_psi,
        )
    return e_total


def make_bonded_all_fused_op(
    *,
    has_box: bool = False,
    dist_eps: float = 1e-10,
    b0_is_ji: bool = True,
    angle_regularization_mode: str = "legacy_denom_eps",
    coulomb_constant: float = 332.0637,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    cmap_first_is_psi: bool = False,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
) -> xtrax.Op:
    """Build an ``xtrax.Op`` for the single-launch fused bonded megakernel (O2.2)."""
    hb = bool(has_box)
    de = float(dist_eps)
    b0_ji = bool(b0_is_ji)
    arm = str(angle_regularization_mode)
    cc = float(coulomb_constant)
    sa = float(softcore_alpha)
    se = int(softcore_exponent)
    fpsi = bool(cmap_first_is_psi)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(
        bonded_all_fused_reference,
        has_box=hb,
        dist_eps=de,
        b0_is_ji=b0_ji,
        coulomb_constant=cc,
        softcore_alpha=sa,
        softcore_exponent=se,
        cmap_first_is_psi=fpsi,
    )

    def _make_vk(interp: bool) -> VjpKernel:
        def _fwd(*args):
            return _pallas_bonded_all_fused_fwd(
                *args, hb, de, b0_ji, arm, cc, sa, se, fpsi, interp, cpg
            )

        def _bwd(res, ct):
            return _pallas_bonded_all_fused_bwd(
                hb, de, b0_ji, arm, cc, sa, se, fpsi, interp, cpg, res, ct
            )

        return VjpKernel(_fwd, _bwd)

    return xtrax.op(
        ref,
        vjp_kernels={"gpu": _make_vk(False), "cpu": _make_vk(True)},
        name="pallas_bonded_all_fused",
        context=context,
    )


def _flatten_multi_term_torsion(
    positions: jax.Array,
    indices: jax.Array,
    params: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_dih = indices.shape[-2]
    is_multi_term = (params.ndim == 4) or (
        params.ndim == 3
        and not (
            positions.ndim == 3
            and params.shape[0] == positions.shape[0]
            and params.shape[1] == n_dih
        )
    )
    if is_multi_term:
        T = params.shape[-2]
        P = params.shape[-1]
        indices = jnp.repeat(indices, T, axis=-2)
        params = params.reshape(*params.shape[:-3], params.shape[-3] * T, P)
        mask = jnp.repeat(mask, T, axis=-1)
    return indices, params, mask


def pallas_bonded_all_fused(
    positions: jax.Array,
    *,
    bond_indices: jax.Array | None = None,
    bonds: jax.Array | None = None,
    bond_params: jax.Array | None = None,
    bond_mask: jax.Array | None = None,
    urey_bradley_indices: jax.Array | None = None,
    urey_bradley_bonds: jax.Array | None = None,
    urey_bradley_params: jax.Array | None = None,
    urey_bradley_mask: jax.Array | None = None,
    angle_indices: jax.Array | None = None,
    angles: jax.Array | None = None,
    angle_params: jax.Array | None = None,
    angle_mask: jax.Array | None = None,
    dihedral_indices: jax.Array | None = None,
    dihedrals: jax.Array | None = None,
    dihedral_params: jax.Array | None = None,
    dihedral_mask: jax.Array | None = None,
    improper_indices: jax.Array | None = None,
    impropers: jax.Array | None = None,
    improper_params: jax.Array | None = None,
    improper_mask: jax.Array | None = None,
    impropers_are_harmonic: bool | None = None,
    improper_is_harmonic: bool | None = None,
    harm_improper_indices: jax.Array | None = None,
    harm_improper_params: jax.Array | None = None,
    harm_improper_mask: jax.Array | None = None,
    exception_pairs: jax.Array | None = None,
    exception_sigmas: jax.Array | None = None,
    exception_epsilons: jax.Array | None = None,
    exception_chargeprods: jax.Array | None = None,
    exception_mask: jax.Array | None = None,
    cmap_torsions: jax.Array | None = None,
    cmap_map_indices: jax.Array | None = None,
    cmap_indices: jax.Array | None = None,
    cmap_coeffs: jax.Array | None = None,
    cmap_mask: jax.Array | None = None,
    cmap_first_is_psi: bool = False,
    box: jax.Array | None = None,
    dist_eps: float = 1e-10,
    b0_is_ji: bool = True,
    angle_regularization_mode: str = "legacy_denom_eps",
    coulomb_constant: float = 332.0637,
    soft_core_lambda: float | jax.Array | None = None,
    softcore_lambda: float | jax.Array | None = None,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    compute_param_grads: bool | None = None,
    return_forces: bool = False,
    interpret: bool | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Single-launch fused bonded Pallas megakernel across all 6 bonded groups (O2.2)."""
    dtype = positions.dtype
    if bond_indices is None:
        bond_indices = bonds
    if urey_bradley_indices is None:
        urey_bradley_indices = urey_bradley_bonds
    if angle_indices is None:
        angle_indices = angles
    if dihedral_indices is None:
        dihedral_indices = dihedrals
    if improper_indices is None:
        improper_indices = impropers
    if impropers_are_harmonic is None:
        impropers_are_harmonic = improper_is_harmonic
    if cmap_map_indices is None:
        cmap_map_indices = cmap_indices

    # 1. Consolidate 2-body harmonic bonds + Urey-Bradley
    b_idx_parts, b_prm_parts, b_msk_parts = [], [], []
    if bond_indices is not None and bond_params is not None and bond_indices.shape[-2] > 0:
        bi = jnp.asarray(bond_indices[..., :, :2], dtype=jnp.int32)
        bp = jnp.asarray(bond_params, dtype=dtype)
        bm = jnp.ones(bi.shape[:-1], dtype=dtype) if bond_mask is None else jnp.asarray(bond_mask, dtype=dtype)
        b_idx_parts.append(bi)
        b_prm_parts.append(bp)
        b_msk_parts.append(bm)
    if urey_bradley_indices is not None and urey_bradley_params is not None and urey_bradley_indices.shape[-2] > 0:
        ui = jnp.asarray(urey_bradley_indices[..., :, :2], dtype=jnp.int32)
        up = jnp.asarray(urey_bradley_params, dtype=dtype)
        um = jnp.ones(ui.shape[:-1], dtype=dtype) if urey_bradley_mask is None else jnp.asarray(urey_bradley_mask, dtype=dtype)
        b_idx_parts.append(ui)
        b_prm_parts.append(up)
        b_msk_parts.append(um)
    if b_idx_parts:
        b_idx_arr = jnp.concatenate(b_idx_parts, axis=-2) if len(b_idx_parts) > 1 else b_idx_parts[0]
        b_prm_arr = jnp.concatenate(b_prm_parts, axis=-2) if len(b_prm_parts) > 1 else b_prm_parts[0]
        b_msk_arr = jnp.concatenate(b_msk_parts, axis=-1) if len(b_msk_parts) > 1 else b_msk_parts[0]
    else:
        b_idx_arr = jnp.zeros((0, 2), dtype=jnp.int32)
        b_prm_arr = jnp.zeros((0, 2), dtype=dtype)
        b_msk_arr = jnp.zeros((0,), dtype=dtype)

    # 2. Harmonic angles
    if angle_indices is not None and angle_params is not None and angle_indices.shape[-2] > 0:
        a_idx_arr = jnp.asarray(angle_indices[..., :, :3], dtype=jnp.int32)
        a_prm_arr = jnp.asarray(angle_params, dtype=dtype)
        a_msk_arr = jnp.ones(a_idx_arr.shape[:-1], dtype=dtype) if angle_mask is None else jnp.asarray(angle_mask, dtype=dtype)
    else:
        a_idx_arr = jnp.zeros((0, 3), dtype=jnp.int32)
        a_prm_arr = jnp.zeros((0, 2), dtype=dtype)
        a_msk_arr = jnp.zeros((0,), dtype=dtype)

    # 3 & 4. Consolidate periodic dihedrals + periodic impropers vs harmonic impropers
    d_idx_parts, d_prm_parts, d_msk_parts = [], [], []
    h_idx_parts, h_prm_parts, h_msk_parts = [], [], []
    if dihedral_indices is not None and dihedral_params is not None and dihedral_indices.shape[-2] > 0:
        di = jnp.asarray(dihedral_indices[..., :, :4], dtype=jnp.int32)
        dp = jnp.asarray(dihedral_params, dtype=dtype)
        dm = jnp.ones(di.shape[:-1], dtype=dtype) if dihedral_mask is None else jnp.asarray(dihedral_mask, dtype=dtype)
        di, dp, dm = _flatten_multi_term_torsion(positions, di, dp, dm)
        d_idx_parts.append(di)
        d_prm_parts.append(dp)
        d_msk_parts.append(dm)
    if improper_indices is not None and improper_params is not None and improper_indices.shape[-2] > 0:
        ii = jnp.asarray(improper_indices[..., :, :4], dtype=jnp.int32)
        ip = jnp.asarray(improper_params, dtype=dtype)
        im = jnp.ones(ii.shape[:-1], dtype=dtype) if improper_mask is None else jnp.asarray(improper_mask, dtype=dtype)
        ii, ip, im = _flatten_multi_term_torsion(positions, ii, ip, im)
        is_harm_imp = bool(impropers_are_harmonic) if impropers_are_harmonic is not None else (ip.shape[-1] < 3)
        if is_harm_imp:
            h_idx_parts.append(ii)
            h_prm_parts.append(ip[..., :2])
            h_msk_parts.append(im)
        else:
            d_idx_parts.append(ii)
            d_prm_parts.append(ip[..., :3])
            d_msk_parts.append(im)
    if harm_improper_indices is not None and harm_improper_params is not None and harm_improper_indices.shape[-2] > 0:
        hi = jnp.asarray(harm_improper_indices[..., :, :4], dtype=jnp.int32)
        hp = jnp.asarray(harm_improper_params, dtype=dtype)
        hm = jnp.ones(hi.shape[:-1], dtype=dtype) if harm_improper_mask is None else jnp.asarray(harm_improper_mask, dtype=dtype)
        hi, hp, hm = _flatten_multi_term_torsion(positions, hi, hp, hm)
        h_idx_parts.append(hi)
        h_prm_parts.append(hp[..., :2])
        h_msk_parts.append(hm)

    if d_idx_parts:
        d_idx_arr = jnp.concatenate(d_idx_parts, axis=-2) if len(d_idx_parts) > 1 else d_idx_parts[0]
        d_prm_arr = jnp.concatenate(d_prm_parts, axis=-2) if len(d_prm_parts) > 1 else d_prm_parts[0]
        d_msk_arr = jnp.concatenate(d_msk_parts, axis=-1) if len(d_msk_parts) > 1 else d_msk_parts[0]
    else:
        d_idx_arr = jnp.zeros((0, 4), dtype=jnp.int32)
        d_prm_arr = jnp.zeros((0, 3), dtype=dtype)
        d_msk_arr = jnp.zeros((0,), dtype=dtype)

    if h_idx_parts:
        h_idx_arr = jnp.concatenate(h_idx_parts, axis=-2) if len(h_idx_parts) > 1 else h_idx_parts[0]
        h_prm_arr = jnp.concatenate(h_prm_parts, axis=-2) if len(h_prm_parts) > 1 else h_prm_parts[0]
        h_msk_arr = jnp.concatenate(h_msk_parts, axis=-1) if len(h_msk_parts) > 1 else h_msk_parts[0]
    else:
        h_idx_arr = jnp.zeros((0, 4), dtype=jnp.int32)
        h_prm_arr = jnp.zeros((0, 2), dtype=dtype)
        h_msk_arr = jnp.zeros((0,), dtype=dtype)

    # 5. 1-4 exceptions
    if (
        exception_pairs is not None
        and exception_sigmas is not None
        and exception_epsilons is not None
        and exception_chargeprods is not None
        and exception_pairs.shape[-2] > 0
    ):
        e_pairs_arr = jnp.asarray(exception_pairs[..., :, :2], dtype=jnp.int32)
        e_sig_arr = jnp.asarray(exception_sigmas, dtype=dtype)
        e_eps_arr = jnp.asarray(exception_epsilons, dtype=dtype)
        e_qp_arr = jnp.asarray(exception_chargeprods, dtype=dtype)
        e_msk_arr = (
            jnp.ones(e_pairs_arr.shape[:-1], dtype=dtype)
            if exception_mask is None
            else jnp.asarray(exception_mask, dtype=dtype)
        )
    else:
        e_pairs_arr = jnp.zeros((0, 2), dtype=jnp.int32)
        e_sig_arr = jnp.zeros((0,), dtype=dtype)
        e_eps_arr = jnp.zeros((0,), dtype=dtype)
        e_qp_arr = jnp.zeros((0,), dtype=dtype)
        e_msk_arr = jnp.zeros((0,), dtype=dtype)

    # 6. CMAP 5-body torsions
    if (
        cmap_torsions is not None
        and cmap_coeffs is not None
        and cmap_torsions.shape[-2] > 0
        and cmap_coeffs.size > 0
        and cmap_coeffs.ndim >= 3
    ):
        if cmap_torsions.shape[-1] == 8:
            c_tors_arr = jnp.concatenate(
                [cmap_torsions[..., :, :4], cmap_torsions[..., :, 7:8]], axis=-1
            ).astype(jnp.int32)
        else:
            c_tors_arr = jnp.asarray(cmap_torsions[..., :, :5], dtype=jnp.int32)
        c_midx_arr = (
            jnp.zeros(c_tors_arr.shape[:-1], dtype=jnp.int32)
            if cmap_map_indices is None
            else jnp.asarray(cmap_map_indices, dtype=jnp.int32)
        )
        c_msk_arr = (
            jnp.ones(c_tors_arr.shape[:-1], dtype=dtype)
            if cmap_mask is None
            else jnp.asarray(cmap_mask, dtype=dtype)
        )
        c_coeffs_arr = jnp.asarray(cmap_coeffs, dtype=dtype)
        if c_coeffs_arr.shape[-1] != 16:
            if c_coeffs_arr.ndim == 3:
                c_coeffs_arr = precompute_cmap_coefficients(c_coeffs_arr).astype(dtype)
            elif c_coeffs_arr.ndim == 4:
                c_coeffs_arr = jax.vmap(precompute_cmap_coefficients)(c_coeffs_arr).astype(dtype)
    else:
        c_tors_arr = jnp.zeros((0, 5), dtype=jnp.int32)
        c_midx_arr = jnp.zeros((0,), dtype=jnp.int32)
        c_msk_arr = jnp.zeros((0,), dtype=dtype)
        c_coeffs_arr = jnp.zeros((0, 24, 24, 16), dtype=dtype)

    eff_lam = soft_core_lambda if soft_core_lambda is not None else (softcore_lambda if softcore_lambda is not None else 1.0)
    lam_arr = jnp.asarray(eff_lam, dtype=dtype)
    box_arr, has_box = _normalize_box(box, positions)

    all_op_args = (
        positions,
        b_idx_arr,
        b_prm_arr,
        b_msk_arr,
        a_idx_arr,
        a_prm_arr,
        a_msk_arr,
        d_idx_arr,
        d_prm_arr,
        d_msk_arr,
        h_idx_arr,
        h_prm_arr,
        h_msk_arr,
        e_pairs_arr,
        e_sig_arr,
        e_eps_arr,
        e_qp_arr,
        e_msk_arr,
        c_tors_arr,
        c_midx_arr,
        c_coeffs_arr,
        c_msk_arr,
        box_arr,
        lam_arr,
    )

    if return_forces:
        if interpret is None:
            interpret = not _triton_gpu_available()
        return _pallas_bonded_all_fused_vmappable(
            *all_op_args,
            bool(has_box),
            float(dist_eps),
            bool(b0_is_ji),
            str(angle_regularization_mode),
            float(coulomb_constant),
            float(softcore_alpha),
            int(softcore_exponent),
            bool(cmap_first_is_psi),
            bool(interpret),
        )

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_bonded_all_fused_op(
            has_box=bool(has_box),
            dist_eps=float(dist_eps),
            b0_is_ji=bool(b0_is_ji),
            angle_regularization_mode=str(angle_regularization_mode),
            coulomb_constant=float(coulomb_constant),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            cmap_first_is_psi=bool(cmap_first_is_psi),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(*all_op_args)

    if interpret is None:
        interpret = not _triton_gpu_available()
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    return _pallas_bonded_all_fused_diff(
        *all_op_args,
        bool(has_box),
        float(dist_eps),
        bool(b0_is_ji),
        str(angle_regularization_mode),
        float(coulomb_constant),
        float(softcore_alpha),
        int(softcore_exponent),
        bool(cmap_first_is_psi),
        bool(interpret),
        cpg,
    )


__all__ = [
    "_load_box6",
    "_min_image3",
    "_min_image_1d",
    "_min_image_host",
    "_normalize_box",
    "_pl_atomic_add_1d",
    "_triton_erf",
    "_triton_erfc",
    "_triton_gpu_available",
    "angles_harmonic_reference",
    "bonded_all_fused_reference",
    "bonds_2body_reference",
    "cmap_torsions_reference",
    "exceptions_14_reference",
    "make_angles_harmonic_op",
    "make_bonded_all_fused_op",
    "make_bonds_2body_op",
    "make_cmap_torsions_op",
    "make_exceptions_14_op",
    "make_pme_exclusion_pairs_op",
    "make_torsions_4body_op",
    "pallas_angles_harmonic",
    "pallas_bonded_all_fused",
    "pallas_bonds_2body",
    "pallas_cmap_torsions",
    "pallas_exceptions_14",
    "pallas_pme_exclusion_pairs",
    "pallas_torsions_4body",
    "pme_exclusion_pairs_reference",
    "torsions_4body_reference",
]
