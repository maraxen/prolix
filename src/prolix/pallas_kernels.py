"""Pallas kernels for GB/Coulomb and explicit-solvent nonbonded physics with full JAX composability.

Provides:
  - ``PallasKernelConfig`` and ``resolve_pallas_mode`` (post-Equinox pure-array boundary & export context)
  - ``gb_coulomb_energy_dense`` and ``gb_energy_split_cutoff`` (with exact position, charge, and Born-radii VJPs)
  - ``pallas_born_radii``: vectorized (32, 32) OBC-II Born radii kernel with custom_vjp and custom_vmap
  - ``pallas_nonbonded_nl``: fused O(N*K) LJ + PME direct-space neighbor-list Pallas kernel with forward-force custom_vjp and custom_vmap
  - ``pallas_flash_explicit_nonbonded``: 2D-tiled (32, 32) dense FlashMD nonbonded Pallas kernel with forward-force custom_vjp and custom_vmap
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.scipy.special import erf, erfc
from proxide.physics import constants

from prolix.export import in_export_context
from prolix.physics.generalized_born import (
    ALPHA_OBC,
    BETA_OBC,
    GAMMA_OBC,
    compute_born_radii_neighbor_list,
    compute_pair_integral,
)
from prolix.physics.neighbor_list import get_neighbor_exclusion_scales

# =============================================================================
# R1: Post-Equinox Kernel Configuration & Backend Resolution
# =============================================================================


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PallasKernelConfig:
    """Configuration for Pallas kernels using standard JAX registered dataclass (zero Equinox)."""

    block_n: int = field(default=32, metadata={"static": True})
    block_k: int = field(default=32, metadata={"static": True})
    use_pallas: bool = field(default=False, metadata={"static": True})
    interpret: bool = field(default=False, metadata={"static": True})


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


def resolve_pallas_mode(
    use_pallas: bool | None = None,
    env_var: str = "PROLIX_USE_PALLAS",
) -> tuple[bool, bool]:
    """Return ``(enabled, interpret)`` for Pallas kernel dispatch.

    Inside ``prolix.export.export_context()``, unconditionally returns
    ``(False, False)`` so ``jax.export`` / IREE WASM always lowers pure-JAX HLO.
    """
    if in_export_context():
        return (False, False)

    if use_pallas is not None:
        enabled = bool(use_pallas)
    else:
        raw = os.environ.get(env_var, "")
        if not raw and env_var != "PROLIX_USE_PALLAS":
            raw = os.environ.get("PROLIX_USE_PALLAS", "")
        enabled = raw.strip().lower() in ("1", "true", "yes", "on")

    if not enabled:
        return (False, False)

    interpret = not _triton_gpu_available()
    return (True, interpret)


# =============================================================================
# R2.2: Tile-based Dense GB+Coulomb (forward + exact analytical cotangents)
# =============================================================================


def _gb_coulomb_energy_chunked(
    positions: jax.Array,     # (N, 3)
    charges: jax.Array,       # (N,)
    born_radii: jax.Array,    # (N,)
    atom_mask: jax.Array,     # (N,) bool
    chunk_size: int = 256,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
) -> jax.Array:
    """Dense N² GB+Coulomb energy WITHOUT materializing (N, N) arrays."""
    N = positions.shape[0]
    dtype = positions.dtype
    charges = charges.astype(dtype)
    born_radii = born_radii.astype(dtype)
    tau = jnp.array(
        (1.0 / solute_dielectric) - (1.0 / solvent_dielectric), dtype=dtype
    )
    prefactor = jnp.array(-0.5 * constants.COULOMB_CONSTANT, dtype=dtype) * tau

    n_chunks = (N + chunk_size - 1) // chunk_size
    N_padded = n_chunks * chunk_size
    pad_n = N_padded - N

    if pad_n > 0:
        positions = jnp.pad(positions, ((0, pad_n), (0, 0)))
        charges = jnp.pad(charges, (0, pad_n))
        born_radii = jnp.pad(born_radii, (0, pad_n), constant_values=1.0)
        atom_mask = jnp.pad(atom_mask, (0, pad_n), constant_values=False)

    eps_small = jnp.array(1e-30, dtype=dtype)
    eps_sqrt = jnp.array(1e-12, dtype=dtype)
    four = jnp.array(4.0, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)

    def chunk_energy(i, acc):
        i_start = i * chunk_size
        pos_i = jax.lax.dynamic_slice(positions, (i_start, 0), (chunk_size, 3))
        q_i = jax.lax.dynamic_slice(charges, (i_start,), (chunk_size,))
        br_i = jax.lax.dynamic_slice(born_radii, (i_start,), (chunk_size,))
        mask_i = jax.lax.dynamic_slice(atom_mask, (i_start,), (chunk_size,))

        delta = pos_i[:, None, :] - positions[None, :, :]
        r_sq = jnp.sum(delta ** 2, axis=-1)

        radii_prod = br_i[:, None] * born_radii[None, :]
        safe_prod = jnp.maximum(radii_prod, eps_small)
        exp_term = jnp.exp(-r_sq / (four * safe_prod))
        f_gb_val = jnp.sqrt(r_sq + radii_prod * exp_term + eps_sqrt)

        charge_prod = q_i[:, None] * charges[None, :]
        energy = charge_prod / f_gb_val

        pair_mask = mask_i[:, None] & atom_mask[None, :]
        energy = jnp.where(pair_mask, energy, zero)

        return (acc + jnp.sum(energy)).astype(dtype)

    total = jax.lax.fori_loop(0, n_chunks, chunk_energy, jnp.zeros((), dtype=dtype))
    return (prefactor * total).astype(dtype)


def _gb_coulomb_cotangents_chunked(
    positions: jax.Array,     # (N, 3)
    charges: jax.Array,       # (N,)
    born_radii: jax.Array,    # (N,)
    atom_mask: jax.Array,     # (N,) bool
    chunk_size: int = 256,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute exact (dE/d_positions, dE/d_charges, dE/d_born_radii) via chunked accumulation.

    Because E_GB = prefactor * sum_{i,j} m_ij * q_i * q_j / f_gb(r_ij, B_i, B_j)
    is symmetric in (i, j), differentiating wrt atom i receives equal contributions
    from row i and column i:
      - dE/dr_i = -2 * prefactor * sum_j m_ij * q_i * q_j * (1 - exp_term / 4) / f_gb^3 * (r_i - r_j)
      - dE/dq_i =  2 * prefactor * sum_j m_ij * q_j / f_gb
      - dE/dB_i = -1 * prefactor * sum_j m_ij * q_i * q_j * B_j * exp_term * (1 + r_sq / (4 * B_i * B_j)) / f_gb^3
    """
    N = positions.shape[0]
    N_orig = N
    dtype = positions.dtype
    charges = charges.astype(dtype)
    born_radii = born_radii.astype(dtype)
    tau = jnp.array(
        (1.0 / solute_dielectric) - (1.0 / solvent_dielectric), dtype=dtype
    )
    prefactor = jnp.array(-0.5 * constants.COULOMB_CONSTANT, dtype=dtype) * tau
    two_prefactor = jnp.array(2.0, dtype=dtype) * prefactor

    n_chunks = (N + chunk_size - 1) // chunk_size
    N_padded = n_chunks * chunk_size
    pad_n = N_padded - N

    if pad_n > 0:
        positions = jnp.pad(positions, ((0, pad_n), (0, 0)))
        charges = jnp.pad(charges, (0, pad_n))
        born_radii = jnp.pad(born_radii, (0, pad_n), constant_values=1.0)
        atom_mask = jnp.pad(atom_mask, (0, pad_n), constant_values=False)

    eps_small = jnp.array(1e-30, dtype=dtype)
    eps_sqrt = jnp.array(1e-12, dtype=dtype)
    four = jnp.array(4.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)

    def chunk_body(i, carry):
        grad_pos_acc, grad_q_acc, grad_br_acc = carry
        i_start = i * chunk_size

        pos_i = jax.lax.dynamic_slice(positions, (i_start, 0), (chunk_size, 3))
        q_i = jax.lax.dynamic_slice(charges, (i_start,), (chunk_size,))
        br_i = jax.lax.dynamic_slice(born_radii, (i_start,), (chunk_size,))
        mask_i = jax.lax.dynamic_slice(atom_mask, (i_start,), (chunk_size,))

        delta = pos_i[:, None, :] - positions[None, :, :]
        r_sq = jnp.sum(delta ** 2, axis=-1)

        radii_prod = br_i[:, None] * born_radii[None, :]
        safe_prod = jnp.maximum(radii_prod, eps_small)
        u_exp = r_sq / (four * safe_prod)
        exp_term = jnp.exp(-u_exp)
        f_gb_sq = r_sq + radii_prod * exp_term + eps_sqrt
        f_gb_val = jnp.sqrt(f_gb_sq)
        f_gb_cubed = f_gb_val * f_gb_sq + eps_small

        pair_mask = mask_i[:, None] & atom_mask[None, :]
        charge_prod = q_i[:, None] * charges[None, :]

        # 1. Position gradient at fixed Born radii:
        # d(1/f_gb)/dr_i = -(1 - exp_term / 4) / f_gb^3 * delta_ij
        # Factor of 2 from (i, j) and (j, i) symmetry in the double sum.
        pos_coeff = jnp.where(
            pair_mask,
            charge_prod * (-(one - exp_term / four) / f_gb_cubed),
            zero,
        )
        grad_pos_i = (two_prefactor * jnp.sum(pos_coeff[:, :, None] * delta, axis=1)).astype(dtype)

        # 2. Charge gradient: dE/dq_i = 2 * prefactor * sum_j m_ij * q_j / f_gb
        q_term = jnp.where(pair_mask, charges[None, :] / f_gb_val, zero)
        grad_q_i = (two_prefactor * jnp.sum(q_term, axis=1)).astype(dtype)

        # 3. Born radius gradient:
        # df_gb / dB_i = (B_j * exp_term * (1 + r_sq / (4 * B_i * B_j))) / (2 * f_gb)
        # Summing row i and column i cancels the 1/2 in df_gb/dB_i against the 2x symmetry!
        br_term = jnp.where(
            pair_mask,
            charge_prod * born_radii[None, :] * exp_term * (one + u_exp) / f_gb_cubed,
            zero,
        )
        grad_br_i = (-prefactor * jnp.sum(br_term, axis=1)).astype(dtype)

        grad_pos_acc = jax.lax.dynamic_update_slice(grad_pos_acc, grad_pos_i, (i_start, 0))
        grad_q_acc = jax.lax.dynamic_update_slice(grad_q_acc, grad_q_i, (i_start,))
        grad_br_acc = jax.lax.dynamic_update_slice(grad_br_acc, grad_br_i, (i_start,))
        return (grad_pos_acc, grad_q_acc, grad_br_acc)

    init_carry = (
        jnp.zeros((N_padded, 3), dtype=dtype),
        jnp.zeros((N_padded,), dtype=dtype),
        jnp.zeros((N_padded,), dtype=dtype),
    )
    grad_pos, grad_q, grad_br = jax.lax.fori_loop(0, n_chunks, chunk_body, init_carry)
    return grad_pos[:N_orig], grad_q[:N_orig], grad_br[:N_orig]


def _gb_coulomb_forces_chunked(
    positions: jax.Array,     # (N, 3)
    charges: jax.Array,       # (N,)
    born_radii: jax.Array,    # (N,)
    atom_mask: jax.Array,     # (N,) bool
    chunk_size: int = 256,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
) -> jax.Array:
    """Dense N² GB+Coulomb analytical position gradient dE/dr_i at fixed Born radii."""
    grad_pos, _, _ = _gb_coulomb_cotangents_chunked(
        positions,
        charges,
        born_radii,
        atom_mask,
        chunk_size=chunk_size,
        solvent_dielectric=solvent_dielectric,
        solute_dielectric=solute_dielectric,
    )
    return grad_pos


@custom_vjp
def gb_coulomb_energy_dense(
    positions: jax.Array,
    charges: jax.Array,
    born_radii: jax.Array,
    atom_mask: jax.Array,
) -> jax.Array:
    """Dense N² GB+Coulomb energy, composable with grad/vmap/pmap."""
    return _gb_coulomb_energy_chunked(positions, charges, born_radii, atom_mask)


def _gb_coulomb_fwd(positions, charges, born_radii, atom_mask):
    energy = _gb_coulomb_energy_chunked(positions, charges, born_radii, atom_mask)
    residuals = (positions, charges, born_radii, atom_mask)
    return energy, residuals


def _gb_coulomb_bwd(residuals, g):
    positions, charges, born_radii, atom_mask = residuals
    grad_positions, grad_charges, grad_born_radii = _gb_coulomb_cotangents_chunked(
        positions, charges, born_radii, atom_mask
    )
    return (
        g * grad_positions,
        g * grad_charges,
        g * grad_born_radii,
        None,
    )


gb_coulomb_energy_dense.defvjp(_gb_coulomb_fwd, _gb_coulomb_bwd)


def gb_energy_split_cutoff(
    positions: jax.Array,     # (N, 3)
    charges: jax.Array,       # (N,)
    radii: jax.Array,         # (N,) intrinsic radii
    atom_mask: jax.Array,     # (N,) bool
    neighbor_idx: jax.Array,  # (N, K) for Born radii NL
    scaled_radii: jax.Array | None = None,
    dielectric_offset: float = 0.09,
    use_pallas: bool | None = None,
) -> jax.Array:
    """Split-cutoff GB energy: NL Born radii + dense N² Coulomb.

    Fully differentiable wrt (positions, charges, radii, scaled_radii),
    including the dE/dB * dB/dr chain-rule term.
    """
    pallas_enabled, interpret = resolve_pallas_mode(use_pallas=use_pallas)
    if pallas_enabled:
        born_radii = pallas_born_radii(
            positions,
            radii,
            neighbor_idx,
            dielectric_offset=dielectric_offset,
            scaled_radii=scaled_radii,
            interpret=interpret,
        )
    else:
        born_radii = compute_born_radii_neighbor_list(
            positions,
            radii,
            neighbor_idx,
            dielectric_offset=dielectric_offset,
            scaled_radii=scaled_radii,
        )

    return gb_coulomb_energy_dense(positions, charges, born_radii, atom_mask)


# =============================================================================
# R5.3: Vectorized (32, 32) Pallas OBC-II Born Radii Kernel
# =============================================================================

_BORN_BLOCK_N = 32
_BORN_BLOCK_K = 32


def _pallas_born_radii_single(
    positions: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    neighbor_idx: jax.Array,
    dielectric_offset: float,
    interpret: bool,
) -> jax.Array:
    """Single-system vectorized (32, 32) Pallas OBC-II Born radii implementation."""
    N, K = neighbor_idx.shape
    dtype = positions.dtype

    n_blocks = max(1, (N + _BORN_BLOCK_N - 1) // _BORN_BLOCK_N)
    N_pad = n_blocks * _BORN_BLOCK_N
    pad_n = N_pad - N

    k_blocks = max(1, (K + _BORN_BLOCK_K - 1) // _BORN_BLOCK_K)
    K_pad = k_blocks * _BORN_BLOCK_K
    pad_k = K_pad - K

    if pad_n > 0:
        pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0)))
        rad_pad = jnp.pad(radii, (0, pad_n), constant_values=1.0)
        sc_rad_pad = jnp.pad(scaled_radii, (0, pad_n), constant_values=1.0)
    else:
        pos_pad = positions
        rad_pad = radii
        sc_rad_pad = scaled_radii

    nbr_pad = neighbor_idx.astype(jnp.int32)
    if pad_n > 0 or pad_k > 0:
        nbr_pad = jnp.pad(nbr_pad, ((0, pad_n), (0, pad_k)), constant_values=N)

    # Pass 1D coordinate columns and flattened neighbor list for 1D NDIndexer compatibility
    pos_x = jnp.asarray(pos_pad[:, 0])
    pos_y = jnp.asarray(pos_pad[:, 1])
    pos_z = jnp.asarray(pos_pad[:, 2])
    nbr_flat = jnp.asarray(nbr_pad.reshape(N_pad * K_pad))

    def _born_kernel(
        px_ref,
        py_ref,
        pz_ref,
        rad_ref,
        sc_rad_ref,
        nbr_flat_ref,
        out_rad_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _BORN_BLOCK_N + jnp.arange(_BORN_BLOCK_N, dtype=jnp.int32)
        is_real_i = offs_i < N

        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        rad_i = jnp.maximum(_pl_load(rad_ref, (offs_i,)), jnp.asarray(1e-6, dtype=dtype))
        offset_rad_i = jnp.maximum(
            rad_i - jnp.asarray(dielectric_offset, dtype=dtype),
            jnp.asarray(1e-4, dtype=dtype),
        )

        def _k_tile_body(k_tile, acc_integral):
            offs_k = k_tile * _BORN_BLOCK_K + jnp.arange(_BORN_BLOCK_K, dtype=jnp.int32)
            flat_nbr_idx = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(
                _BORN_BLOCK_N * _BORN_BLOCK_K
            )
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_nbr_idx,))
            j_idx = j_idx_1d.reshape(_BORN_BLOCK_N, _BORN_BLOCK_K)
            valid_nbr = (j_idx >= 0) & (j_idx < N) & is_real_i[:, None]
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_BORN_BLOCK_N, _BORN_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_BORN_BLOCK_N, _BORN_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_BORN_BLOCK_N, _BORN_BLOCK_K)
            sc_rad_j = _pl_load(sc_rad_ref, (safe_j_1d,)).reshape(_BORN_BLOCK_N, _BORN_BLOCK_K)

            dx = px_i[:, None] - px_j
            dy = py_i[:, None] - py_j
            dz = pz_i[:, None] - pz_j
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + jnp.asarray(1e-12, dtype=dtype))

            pair_int = compute_pair_integral(dist, offset_rad_i[:, None], sc_rad_j)
            pair_int = jnp.where(valid_nbr, pair_int, jnp.asarray(0.0, dtype=dtype))
            return acc_integral + jnp.sum(pair_int, axis=1)

        integral_sum = jax.lax.fori_loop(
            0,
            k_blocks,
            _k_tile_body,
            jnp.zeros((_BORN_BLOCK_N,), dtype=dtype),
        )

        scaled_integral = offset_rad_i * integral_sum
        tanh_arg = (
            jnp.asarray(ALPHA_OBC, dtype=dtype) * scaled_integral
            - jnp.asarray(BETA_OBC, dtype=dtype) * (scaled_integral ** 2)
            + jnp.asarray(GAMMA_OBC, dtype=dtype) * (scaled_integral ** 3)
        )
        inv_born = (
            jnp.asarray(1.0, dtype=dtype) / offset_rad_i
            - jnp.tanh(tanh_arg) / jnp.maximum(rad_i, jnp.asarray(1e-4, dtype=dtype))
        )
        born_rad = jnp.asarray(1.0, dtype=dtype) / jnp.maximum(
            inv_born, jnp.asarray(1e-4, dtype=dtype)
        )
        born_rad = jnp.where(is_real_i, born_rad, jnp.asarray(0.0, dtype=dtype))
        _pl_store(out_rad_ref, (offs_i,), born_rad)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(6)]
    out_specs = pl.BlockSpec(memory_space=pl.ANY)
    out_shape = jax.ShapeDtypeStruct((N_pad,), dtype)

    born_padded = pl.pallas_call(
        _born_kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_born_radii",
    )(pos_x, pos_y, pos_z, rad_pad, sc_rad_pad, nbr_flat)
    return born_padded[:N]


def _pallas_born_radii_vmappable(
    positions: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    neighbor_idx: jax.Array,
    dielectric_offset: float,
    interpret: bool,
) -> jax.Array:
    d_off = float(dielectric_offset)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, rad, sc_rad, nbr):
        return _pallas_born_radii_single(pos, rad, sc_rad, nbr, d_off, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, rad, sc_rad, nbr):
        def _step(b_idx):
            p = pos[b_idx] if in_batched[0] else pos
            r = rad[b_idx] if in_batched[1] else rad
            sr = sc_rad[b_idx] if in_batched[2] else sc_rad
            nb = nbr[b_idx] if in_batched[3] else nbr
            return _vmappable(p, r, sr, nb)

        out = jax.lax.map(_step, jnp.arange(axis_size))
        return out, True

    return _vmappable(positions, radii, scaled_radii, neighbor_idx)


@functools.partial(custom_vjp, nondiff_argnums=(3, 5))
def _pallas_born_radii_diff(
    positions: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    dielectric_offset: float,
    neighbor_idx: jax.Array,
    interpret: bool,
) -> jax.Array:
    return _pallas_born_radii_vmappable(
        positions, radii, scaled_radii, neighbor_idx, dielectric_offset, interpret
    )


def _pallas_born_radii_fwd(positions, radii, scaled_radii, dielectric_offset, neighbor_idx, interpret):
    out = _pallas_born_radii_vmappable(
        positions, radii, scaled_radii, neighbor_idx, dielectric_offset, interpret
    )
    return out, (positions, radii, scaled_radii, neighbor_idx)


def _pallas_born_radii_bwd(dielectric_offset, interpret, res, g):
    positions, radii, scaled_radii, neighbor_idx = res

    def _ref_fn(pos, rad, sc_rad):
        return compute_born_radii_neighbor_list(
            pos,
            rad,
            neighbor_idx,
            dielectric_offset=dielectric_offset,
            scaled_radii=sc_rad,
        )

    _, vjp_fn = jax.vjp(_ref_fn, positions, radii, scaled_radii)
    grad_pos, grad_rad, grad_sc_rad = vjp_fn(g)
    return (grad_pos, grad_rad, grad_sc_rad, jnp.zeros_like(neighbor_idx))


_pallas_born_radii_diff.defvjp(_pallas_born_radii_fwd, _pallas_born_radii_bwd)


def pallas_born_radii(
    positions: jax.Array,
    radii: jax.Array,
    neighbor_idx: jax.Array,
    dielectric_offset: float = 0.09,
    scaled_radii: jax.Array | None = None,
    interpret: bool | None = None,
) -> jax.Array:
    """Computes OBC-II Born radii using a vectorized (32, 32) Pallas kernel.

    Supports ``jax.grad`` (via ``custom_vjp``), ``jax.vmap`` (via ``custom_vmap``),
    and automatic ``interpret=True`` fallback on CPU.
    """
    if scaled_radii is None:
        scaled_radii = radii
    if interpret is None:
        interpret = not _triton_gpu_available()
    if in_export_context():
        return compute_born_radii_neighbor_list(
            positions,
            radii,
            neighbor_idx,
            dielectric_offset=dielectric_offset,
            scaled_radii=scaled_radii,
        )
    return _pallas_born_radii_diff(
        positions,
        radii,
        scaled_radii,
        float(dielectric_offset),
        neighbor_idx,
        bool(interpret),
    )


# =============================================================================
# R5.1: Fused O(N*K) Neighbor-List Nonbonded Pallas Kernel
# =============================================================================

_NB_BLOCK_N = 32
_NB_BLOCK_K = 32
_NB_MAX_EXCL = 32


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


def _lj_switch_weight_and_deriv(
    dist: jax.Array,
    cutoff: float,
    switch_width: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute C2 smooth switching weight S(r) and derivative S'(r)."""
    dtype = dist.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    r_cut = jnp.asarray(cutoff, dtype=dtype)
    in_cut = (dist < r_cut).astype(dtype)
    if switch_width <= 0.0:
        return in_cut, jnp.zeros_like(dist)

    w = jnp.asarray(switch_width, dtype=dtype)
    r_sw = r_cut - w
    z = jnp.clip((dist - r_sw) / w, zero, one)
    s_poly = one - jnp.asarray(10.0, dtype=dtype) * (z ** 3) + jnp.asarray(15.0, dtype=dtype) * (z ** 4) - jnp.asarray(6.0, dtype=dtype) * (z ** 5)
    s_val = jnp.where(dist <= r_sw, one, jnp.where(dist < r_cut, s_poly, zero))
    ds_dr = jnp.where(
        (dist > r_sw) & (dist < r_cut),
        jnp.asarray(-30.0, dtype=dtype) * (z ** 2) * ((one - z) ** 2) / w,
        zero,
    )
    return s_val, ds_dr


def _pallas_nonbonded_nl_single(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    lj_switch_width: float,
    soft_core_lambda: float,
    include_pme_excl_erf: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """Single-system vectorized (32, 32) Pallas O(N*K) nonbonded energy & forces."""
    N, K = neighbor_idx.shape
    M = excl_indices.shape[1]
    dtype = positions.dtype

    n_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    N_pad = n_blocks * _NB_BLOCK_N
    pad_n = N_pad - N

    k_blocks = max(1, (K + _NB_BLOCK_K - 1) // _NB_BLOCK_K)
    K_pad = k_blocks * _NB_BLOCK_K
    pad_k = K_pad - K

    pad_m = max(0, _NB_MAX_EXCL - M)

    if pad_n > 0:
        pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0)))
        q_pad = jnp.pad(charges, (0, pad_n))
        sig_pad = jnp.pad(sigmas, (0, pad_n), constant_values=1.0)
        eps_pad = jnp.pad(epsilons, (0, pad_n))
        mask_pad = jnp.pad(atom_mask.astype(jnp.int32), (0, pad_n), constant_values=0)
    else:
        pos_pad = positions
        q_pad = charges
        sig_pad = sigmas
        eps_pad = epsilons
        mask_pad = atom_mask.astype(jnp.int32)

    excl_idx_pad = excl_indices[:, :_NB_MAX_EXCL].astype(jnp.int32)
    excl_vdw_pad = excl_scales_vdw[:, :_NB_MAX_EXCL].astype(dtype)
    excl_elec_pad = excl_scales_elec[:, :_NB_MAX_EXCL].astype(dtype)
    if pad_n > 0 or pad_m > 0:
        excl_idx_pad = jnp.pad(excl_idx_pad, ((0, pad_n), (0, pad_m)), constant_values=-1)
        excl_vdw_pad = jnp.pad(excl_vdw_pad, ((0, pad_n), (0, pad_m)), constant_values=1.0)
        excl_elec_pad = jnp.pad(excl_elec_pad, ((0, pad_n), (0, pad_m)), constant_values=1.0)

    nbr_pad = neighbor_idx.astype(jnp.int32)
    if pad_n > 0 or pad_k > 0:
        nbr_pad = jnp.pad(nbr_pad, ((0, pad_n), (0, pad_k)), constant_values=N)

    pos_x = jnp.asarray(pos_pad[:, 0])
    pos_y = jnp.asarray(pos_pad[:, 1])
    pos_z = jnp.asarray(pos_pad[:, 2])
    nbr_flat = jnp.asarray(nbr_pad.reshape(N_pad * K_pad))
    excl_idx_flat = jnp.asarray(excl_idx_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_vdw_flat = jnp.asarray(excl_vdw_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_elec_flat = jnp.asarray(excl_elec_pad.reshape(N_pad * _NB_MAX_EXCL))
    box_vec = jnp.asarray(box.astype(dtype))

    def _nb_nl_kernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        sig_ref,
        eps_ref,
        mask_ref,
        nbr_flat_ref,
        excl_idx_flat_ref,
        excl_vdw_flat_ref,
        excl_elec_flat_ref,
        box_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _NB_BLOCK_N + jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)

        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        q_i = _pl_load(q_ref, (offs_i,))
        sig_i = _pl_load(sig_ref, (offs_i,))
        eps_i = _pl_load(eps_ref, (offs_i,))

        # Preload (32, 32) exclusions for this block of i-atoms
        offs_m = jnp.arange(_NB_MAX_EXCL, dtype=jnp.int32)
        flat_excl_offs = (offs_i[:, None] * _NB_MAX_EXCL + offs_m[None, :]).reshape(
            _NB_BLOCK_N * _NB_MAX_EXCL
        )
        excl_idx_i = _pl_load(excl_idx_flat_ref, (flat_excl_offs,)).reshape(
            _NB_BLOCK_N, _NB_MAX_EXCL
        )
        excl_vdw_i = _pl_load(excl_vdw_flat_ref, (flat_excl_offs,)).reshape(
            _NB_BLOCK_N, _NB_MAX_EXCL
        )
        excl_elec_i = _pl_load(excl_elec_flat_ref, (flat_excl_offs,)).reshape(
            _NB_BLOCK_N, _NB_MAX_EXCL
        )
        valid_excl_i = excl_idx_i >= 0

        bx = _pl_load(box_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (jnp.ones((1,), dtype=jnp.int32),))[0]
        bz = _pl_load(box_ref, (jnp.full((1,), 2, dtype=jnp.int32),))[0]

        lam = jnp.asarray(soft_core_lambda, dtype=dtype)
        soft_alpha_lj = jnp.asarray(0.5, dtype=dtype) * (jnp.asarray(1.0, dtype=dtype) - lam)
        coul_const = jnp.asarray(constants.COULOMB_CONSTANT, dtype=dtype)
        alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
        two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

        def _k_tile_body(k_tile, carry):
            acc_e, acc_fx, acc_fy, acc_fz = carry
            offs_k = k_tile * _NB_BLOCK_K + jnp.arange(_NB_BLOCK_K, dtype=jnp.int32)
            flat_nbr_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(
                _NB_BLOCK_N * _NB_BLOCK_K
            )
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_nbr_offs,))
            j_idx = j_idx_1d.reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            q_j = _pl_load(q_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            sig_j = _pl_load(sig_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            eps_j = _pl_load(eps_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
            mask_j = _pl_load(mask_ref, (safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K) > 0

            valid_pair = (
                mask_i[:, None]
                & mask_j
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
            )

            dx = px_i[:, None] - px_j
            dy = py_i[:, None] - py_j
            dz = pz_i[:, None] - pz_j
            if has_box:
                half = jnp.asarray(0.5, dtype=dtype)
                dx = dx - bx * jnp.floor(dx / bx + half)
                dy = dy - by * jnp.floor(dy / by + half)
                dz = dz - bz * jnp.floor(dz / bz + half)

            r_sq = dx * dx + dy * dy + dz * dz + jnp.asarray(1e-12, dtype=dtype)
            dist = jnp.sqrt(r_sq)
            in_cutoff = dist < jnp.asarray(cutoff, dtype=dtype)
            active = valid_pair & in_cutoff

            # Match j_idx against preloaded (32, 32) exclusion table in registers
            matches = (j_idx[:, :, None] == excl_idx_i[:, None, :]) & valid_excl_i[:, None, :]
            any_match = jnp.sum(matches.astype(jnp.int32), axis=2) > 0
            w_vdw = jnp.where(
                any_match,
                jnp.sum(jnp.where(matches, excl_vdw_i[:, None, :], jnp.asarray(0.0, dtype=dtype)), axis=2),
                jnp.asarray(1.0, dtype=dtype),
            )
            w_elec = jnp.where(
                any_match,
                jnp.sum(jnp.where(matches, excl_elec_i[:, None, :], jnp.asarray(0.0, dtype=dtype)), axis=2),
                jnp.asarray(1.0, dtype=dtype),
            )

            # --- Lennard-Jones ---
            sig_ij = jnp.asarray(0.5, dtype=dtype) * (sig_i[:, None] + sig_j)
            sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), jnp.asarray(1.0, dtype=dtype))
            eps_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j, jnp.asarray(0.0, dtype=dtype)) + jnp.asarray(1e-30, dtype=dtype))

            zero = jnp.asarray(0.0, dtype=dtype)
            u = dist / sig_safe
            u6 = u ** 6
            denom_lj = soft_alpha_lj + u6
            x_inv6 = jnp.asarray(1.0, dtype=dtype) / jnp.maximum(denom_lj, jnp.asarray(1e-6, dtype=dtype))
            v_unsw = jnp.asarray(4.0, dtype=dtype) * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)
            dv_unsw_dr = (
                jnp.asarray(-24.0, dtype=dtype)
                * eps_ij
                * lam
                * (jnp.asarray(2.0, dtype=dtype) * x_inv6 - jnp.asarray(1.0, dtype=dtype))
                * (x_inv6 * x_inv6)
                * u6
                / dist
            )

            s_w, ds_dr = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
            e_lj = w_vdw * v_unsw * s_w
            de_lj_dr = w_vdw * (dv_unsw_dr * s_w + v_unsw * ds_dr)

            # --- Electrostatics ---
            qq = coul_const * q_i[:, None] * q_j
            if pme_alpha > 0.0:
                alpha_r = alpha_val * dist
                exp_term = jnp.exp(zero - (alpha_r * alpha_r))
                if include_pme_excl_erf:
                    eff_w = w_elec - _triton_erf(alpha_r)
                    e_coul = qq * eff_w / dist
                    de_coul_dr = qq * (
                        (zero - eff_w) / r_sq
                        - (alpha_val * two_over_sqrt_pi * exp_term) / dist
                    )
                else:
                    erfc_val = _triton_erfc(alpha_r)
                    e_coul = qq * w_elec * erfc_val / dist
                    de_coul_dr = qq * w_elec * (
                        (zero - erfc_val) / r_sq
                        - (alpha_val * two_over_sqrt_pi * exp_term) / dist
                    )
            else:
                e_coul = qq * w_elec / dist
                de_coul_dr = (zero - qq) * w_elec / r_sq

            e_pair = jnp.where(active, e_lj + e_coul, zero)
            # Force F_i = -grad_{r_i} E = - sum_j (de_pair / dr_ij) * (r_i - r_j) / r_ij
            f_scalar = jnp.where(active, (zero - (de_lj_dr + de_coul_dr)) / dist, zero)

            acc_e = acc_e + jnp.asarray(0.5, dtype=dtype) * jnp.sum(e_pair, axis=1)
            acc_fx = acc_fx + jnp.sum(f_scalar * dx, axis=1)
            acc_fy = acc_fy + jnp.sum(f_scalar * dy, axis=1)
            acc_fz = acc_fz + jnp.sum(f_scalar * dz, axis=1)
            return (acc_e, acc_fx, acc_fy, acc_fz)

        init_carry = (
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
        )
        e_blk, fx_blk, fy_blk, fz_blk = jax.lax.fori_loop(0, k_blocks, _k_tile_body, init_carry)
        _pl_store(e_out_ref, (offs_i,), e_blk)
        _pl_store(fx_out_ref, (offs_i,), fx_blk)
        _pl_store(fy_out_ref, (offs_i,), fy_blk)
        _pl_store(fz_out_ref, (offs_i,), fz_blk)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(12)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [jax.ShapeDtypeStruct((N_pad,), dtype) for _ in range(4)]

    e_pad, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _nb_nl_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_nonbonded_nl",
    )(
        pos_x,
        pos_y,
        pos_z,
        q_pad,
        sig_pad,
        eps_pad,
        mask_pad,
        nbr_flat,
        excl_idx_flat,
        excl_vdw_flat,
        excl_elec_flat,
        box_vec,
    )
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1)
    return jnp.sum(e_pad[:N]), forces


def _pallas_nonbonded_nl_vmappable(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    lj_switch_width: float,
    soft_core_lambda: float,
    include_pme_excl_erf: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    hb = bool(has_box)
    pa = float(pme_alpha)
    co = float(cutoff)
    sw = float(lj_switch_width)
    lam = float(soft_core_lambda)
    inc_erf = bool(include_pme_excl_erf)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, sig, eps, mask, nbr, ei, ev, ee, bx):
        return _pallas_nonbonded_nl_single(
            pos,
            q,
            sig,
            eps,
            mask,
            nbr,
            ei,
            ev,
            ee,
            bx,
            hb,
            pa,
            co,
            sw,
            lam,
            inc_erf,
            interp,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, sig, eps, mask, nbr, ei, ev, ee, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                q[b_idx] if in_batched[1] else q,
                sig[b_idx] if in_batched[2] else sig,
                eps[b_idx] if in_batched[3] else eps,
                mask[b_idx] if in_batched[4] else mask,
                nbr[b_idx] if in_batched[5] else nbr,
                ei[b_idx] if in_batched[6] else ei,
                ev[b_idx] if in_batched[7] else ev,
                ee[b_idx] if in_batched[8] else ee,
                bx[b_idx] if in_batched[9] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    return _vmappable(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
    )


def _nonbonded_nl_param_grads(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    lj_switch_width: float,
    soft_core_lambda: float,
    include_pme_excl_erf: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Analytical parameter gradients (grad_q, grad_sigma, grad_eps) over (N, K)."""

    N = positions.shape[0]
    dtype = positions.dtype
    safe_j = jnp.clip(neighbor_idx, 0, N - 1)
    r_i = positions[:, None, :]
    r_j = positions[safe_j]
    dr = r_i - r_j
    if has_box:
        dr = dr - box[None, None, :] * jnp.floor(
            dr / box[None, None, :] + jnp.asarray(0.5, dtype=dtype)
        )
    r_sq = jnp.sum(dr ** 2, axis=-1) + jnp.asarray(1e-12, dtype=dtype)
    dist = jnp.sqrt(r_sq)

    i_idx = jnp.arange(N, dtype=jnp.int32)[:, None]
    active = (
        atom_mask[:, None]
        & atom_mask[safe_j]
        & (neighbor_idx >= 0)
        & (neighbor_idx < N)
        & (neighbor_idx != i_idx)
        & (dist < jnp.asarray(cutoff, dtype=dtype))
    )

    w_vdw, w_elec = get_neighbor_exclusion_scales(
        excl_indices, excl_scales_vdw, excl_scales_elec, neighbor_idx
    )
    w_vdw = w_vdw.astype(dtype)
    w_elec = w_elec.astype(dtype)

    sig_ij = jnp.asarray(0.5, dtype=dtype) * (sigmas[:, None] + sigmas[safe_j])
    sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), jnp.asarray(1.0, dtype=dtype))
    eps_ij = jnp.sqrt(jnp.maximum(epsilons[:, None] * epsilons[safe_j], jnp.asarray(0.0, dtype=dtype)) + jnp.asarray(1e-30, dtype=dtype))

    lam = jnp.asarray(soft_core_lambda, dtype=dtype)
    soft_alpha_lj = jnp.asarray(0.5, dtype=dtype) * (jnp.asarray(1.0, dtype=dtype) - lam)
    u = dist / sig_safe
    u6 = u ** 6
    x_inv6 = jnp.asarray(1.0, dtype=dtype) / jnp.maximum(soft_alpha_lj + u6, jnp.asarray(1e-6, dtype=dtype))
    s_w, _ = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)

    # dE_LJ / d(sigma_i):
    # de_ij / ds_ij = 24 * eps_ij * lam * (2 * x_inv6 - 1) * x_inv6^2 * u^6 / s_ij
    # ds_ij / d(sigma_i) = 1/2, and summing row i + col i cancels 1/2 from total energy!
    de_ds_ij = (
        jnp.asarray(24.0, dtype=dtype)
        * eps_ij
        * lam
        * (jnp.asarray(2.0, dtype=dtype) * x_inv6 - jnp.asarray(1.0, dtype=dtype))
        * (x_inv6 * x_inv6)
        * u6
        / sig_safe
    )
    grad_sig = jnp.sum(
        jnp.where(active, jnp.asarray(0.5, dtype=dtype) * w_vdw * s_w * de_ds_ij, jnp.asarray(0.0, dtype=dtype)),
        axis=1,
    )

    # dE_LJ / d(eps_i):
    # e_ij = 4 * eps_ij * lam * (x_inv6^2 - x_inv6), d(eps_ij)/d(eps_i) = 0.5 * eps_j / eps_ij
    v_shape = jnp.asarray(4.0, dtype=dtype) * lam * (x_inv6 * x_inv6 - x_inv6)
    deps_ij_depsi = jnp.asarray(0.5, dtype=dtype) * epsilons[safe_j] / jnp.maximum(eps_ij, jnp.asarray(1e-15, dtype=dtype))
    grad_eps = jnp.sum(
        jnp.where(active, w_vdw * s_w * v_shape * deps_ij_depsi, jnp.asarray(0.0, dtype=dtype)),
        axis=1,
    )

    # dE_coul / d(q_i):
    coul_const = jnp.asarray(constants.COULOMB_CONSTANT, dtype=dtype)
    alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
    if pme_alpha > 0.0:
        if include_pme_excl_erf:
            coul_kernel = (w_elec - erf(alpha_val * dist)) / dist
        else:
            coul_kernel = (w_elec * erfc(alpha_val * dist)) / dist
    else:
        coul_kernel = w_elec / dist
    grad_q = jnp.sum(
        jnp.where(active, coul_const * charges[safe_j] * coul_kernel, jnp.asarray(0.0, dtype=dtype)),
        axis=1,
    )
    return grad_q, grad_sig, grad_eps


@functools.partial(custom_vjp, nondiff_argnums=(10, 11, 12, 13, 14, 15, 16))
def _pallas_nonbonded_nl_diff(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    lj_switch_width: float,
    soft_core_lambda: float,
    include_pme_excl_erf: bool,
    interpret: bool,
) -> jax.Array:
    energy, _ = _pallas_nonbonded_nl_vmappable(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        has_box,
        pme_alpha,
        cutoff,
        lj_switch_width,
        soft_core_lambda,
        include_pme_excl_erf,
        interpret,
    )
    return energy


def _pallas_nonbonded_nl_fwd(
    positions,
    charges,
    sigmas,
    epsilons,
    atom_mask,
    neighbor_idx,
    excl_indices,
    excl_scales_vdw,
    excl_scales_elec,
    box,
    has_box,
    pme_alpha,
    cutoff,
    lj_switch_width,
    soft_core_lambda,
    include_pme_excl_erf,
    interpret,
):
    energy, forces = _pallas_nonbonded_nl_vmappable(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        has_box,
        pme_alpha,
        cutoff,
        lj_switch_width,
        soft_core_lambda,
        include_pme_excl_erf,
        interpret,
    )
    res = (
        forces,
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
    )
    return energy, res


def _pallas_nonbonded_nl_bwd(
    has_box,
    pme_alpha,
    cutoff,
    lj_switch_width,
    soft_core_lambda,
    include_pme_excl_erf,
    interpret,
    res,
    g,
):
    (
        forces,
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
    ) = res
    grad_pos = -g[..., None, None] * forces if g.ndim > 0 else -g * forces

    if positions.ndim == 2:
        grad_q, grad_sig, grad_eps = _nonbonded_nl_param_grads(
            positions,
            charges,
            sigmas,
            epsilons,
            atom_mask,
            neighbor_idx,
            excl_indices,
            excl_scales_vdw,
            excl_scales_elec,
            box,
            has_box,
            pme_alpha,
            cutoff,
            lj_switch_width,
            soft_core_lambda,
            include_pme_excl_erf,
        )
        grad_q = g * grad_q
        grad_sig = g * grad_sig
        grad_eps = g * grad_eps
    else:
        grad_q, grad_sig, grad_eps = jax.vmap(
            lambda p, q, s, e, m, nb, ei, ev, ee, b, g_i: tuple(
                g_i * x
                for x in _nonbonded_nl_param_grads(
                    p, q, s, e, m, nb, ei, ev, ee, b,
                    has_box, pme_alpha, cutoff, lj_switch_width, soft_core_lambda, include_pme_excl_erf
                )
            )
        )(
            positions, charges, sigmas, epsilons, atom_mask, neighbor_idx,
            excl_indices, excl_scales_vdw, excl_scales_elec, box, g
        )

    return (
        grad_pos,
        grad_q,
        grad_sig,
        grad_eps,
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(neighbor_idx),
        jnp.zeros_like(excl_indices),
        jnp.zeros_like(excl_scales_vdw),
        jnp.zeros_like(excl_scales_elec),
        jnp.zeros_like(box),
    )


_pallas_nonbonded_nl_diff.defvjp(_pallas_nonbonded_nl_fwd, _pallas_nonbonded_nl_bwd)


def pallas_nonbonded_nl(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array | None = None,
    pme_alpha: float = 0.0,
    cutoff: float = 10.0,
    lj_switch_width: float = 0.0,
    soft_core_lambda: float = 1.0,
    include_pme_excl_erf: bool = True,
    interpret: bool | None = None,
) -> jax.Array:
    """Fused O(N*K) neighbor-list LJ + Coulomb/PME direct-space Pallas kernel."""
    if interpret is None:
        interpret = not _triton_gpu_available()
    has_box = box is not None
    box_arr = (
        jnp.asarray(box, dtype=positions.dtype)
        if box is not None
        else jnp.ones((3,), dtype=positions.dtype)
    )
    if box_arr.ndim == 1 and positions.ndim == 3:
        box_arr = jnp.broadcast_to(box_arr[None, :], (positions.shape[0], 3))

    return _pallas_nonbonded_nl_diff(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box_arr,
        bool(has_box),
        float(pme_alpha),
        float(cutoff),
        float(lj_switch_width),
        float(soft_core_lambda),
        bool(include_pme_excl_erf),
        bool(interpret),
    )


# =============================================================================
# R5.2: 2D-Tiled (32, 32) Flash Explicit Nonbonded Pallas Kernel
# =============================================================================

_FLASH_BLOCK_I = 32
_FLASH_BLOCK_J = 32


def _pallas_flash_explicit_single(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    soft_core_lambda: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """Single-system 2D-tiled (32, 32) Pallas dense explicit nonbonded energy & forces."""
    N = positions.shape[0]
    M = excl_indices.shape[1]
    dtype = positions.dtype

    n_blocks = max(1, (N + _FLASH_BLOCK_I - 1) // _FLASH_BLOCK_I)
    N_pad = n_blocks * _FLASH_BLOCK_I
    pad_n = N_pad - N
    pad_m = max(0, _NB_MAX_EXCL - M)

    if pad_n > 0:
        pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0)))
        q_pad = jnp.pad(charges, (0, pad_n))
        sig_pad = jnp.pad(sigmas, (0, pad_n), constant_values=1.0)
        eps_pad = jnp.pad(epsilons, (0, pad_n))
        mask_pad = jnp.pad(atom_mask.astype(jnp.int32), (0, pad_n), constant_values=0)
    else:
        pos_pad = positions
        q_pad = charges
        sig_pad = sigmas
        eps_pad = epsilons
        mask_pad = atom_mask.astype(jnp.int32)

    excl_idx_pad = excl_indices[:, :_NB_MAX_EXCL].astype(jnp.int32)
    excl_vdw_pad = excl_scales_vdw[:, :_NB_MAX_EXCL].astype(dtype)
    excl_elec_pad = excl_scales_elec[:, :_NB_MAX_EXCL].astype(dtype)
    if pad_n > 0 or pad_m > 0:
        excl_idx_pad = jnp.pad(excl_idx_pad, ((0, pad_n), (0, pad_m)), constant_values=-1)
        excl_vdw_pad = jnp.pad(excl_vdw_pad, ((0, pad_n), (0, pad_m)), constant_values=1.0)
        excl_elec_pad = jnp.pad(excl_elec_pad, ((0, pad_n), (0, pad_m)), constant_values=1.0)

    pos_x = jnp.asarray(pos_pad[:, 0])
    pos_y = jnp.asarray(pos_pad[:, 1])
    pos_z = jnp.asarray(pos_pad[:, 2])
    excl_idx_flat = jnp.asarray(excl_idx_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_vdw_flat = jnp.asarray(excl_vdw_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_elec_flat = jnp.asarray(excl_elec_pad.reshape(N_pad * _NB_MAX_EXCL))
    box_vec = jnp.asarray(box.astype(dtype))

    def _flash_kernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        sig_ref,
        eps_ref,
        mask_ref,
        excl_idx_flat_ref,
        excl_vdw_flat_ref,
        excl_elec_flat_ref,
        box_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _FLASH_BLOCK_I + jnp.arange(_FLASH_BLOCK_I, dtype=jnp.int32)
        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)

        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        q_i = _pl_load(q_ref, (offs_i,))
        sig_i = _pl_load(sig_ref, (offs_i,))
        eps_i = _pl_load(eps_ref, (offs_i,))

        offs_m = jnp.arange(_NB_MAX_EXCL, dtype=jnp.int32)
        flat_excl_offs = (offs_i[:, None] * _NB_MAX_EXCL + offs_m[None, :]).reshape(
            _FLASH_BLOCK_I * _NB_MAX_EXCL
        )
        excl_idx_i = _pl_load(excl_idx_flat_ref, (flat_excl_offs,)).reshape(
            _FLASH_BLOCK_I, _NB_MAX_EXCL
        )
        excl_vdw_i = _pl_load(excl_vdw_flat_ref, (flat_excl_offs,)).reshape(
            _FLASH_BLOCK_I, _NB_MAX_EXCL
        )
        excl_elec_i = _pl_load(excl_elec_flat_ref, (flat_excl_offs,)).reshape(
            _FLASH_BLOCK_I, _NB_MAX_EXCL
        )
        valid_excl_i = excl_idx_i >= 0

        bx = _pl_load(box_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (jnp.ones((1,), dtype=jnp.int32),))[0]
        bz = _pl_load(box_ref, (jnp.full((1,), 2, dtype=jnp.int32),))[0]

        lam = jnp.asarray(soft_core_lambda, dtype=dtype)
        soft_alpha_lj = jnp.asarray(0.5, dtype=dtype) * (jnp.asarray(1.0, dtype=dtype) - lam)
        soft_alpha_coul = jnp.asarray(2.0, dtype=dtype) * (jnp.asarray(1.0, dtype=dtype) - lam)
        coul_const = jnp.asarray(constants.COULOMB_CONSTANT, dtype=dtype)
        alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
        two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

        def _j_tile_body(j_tile, carry):
            acc_e, acc_fx, acc_fy, acc_fz = carry
            offs_j = j_tile * _FLASH_BLOCK_J + jnp.arange(_FLASH_BLOCK_J, dtype=jnp.int32)

            px_j = _pl_load(px_ref, (offs_j,))
            py_j = _pl_load(py_ref, (offs_j,))
            pz_j = _pl_load(pz_ref, (offs_j,))
            q_j = _pl_load(q_ref, (offs_j,))
            sig_j = _pl_load(sig_ref, (offs_j,))
            eps_j = _pl_load(eps_ref, (offs_j,))
            mask_j = (_pl_load(mask_ref, (offs_j,)) > 0) & (offs_j < N)

            active = mask_i[:, None] & mask_j[None, :] & (offs_i[:, None] != offs_j[None, :])

            dx = px_i[:, None] - px_j[None, :]
            dy = py_i[:, None] - py_j[None, :]
            dz = pz_i[:, None] - pz_j[None, :]
            if has_box:
                half = jnp.asarray(0.5, dtype=dtype)
                dx = dx - bx * jnp.floor(dx / bx + half)
                dy = dy - by * jnp.floor(dy / by + half)
                dz = dz - bz * jnp.floor(dz / bz + half)

            dist_sq = dx * dx + dy * dy + dz * dz + jnp.asarray(1e-10, dtype=dtype)
            dist = jnp.sqrt(dist_sq)

            matches = (offs_j[None, :, None] == excl_idx_i[:, None, :]) & valid_excl_i[:, None, :]
            any_match = jnp.sum(matches.astype(jnp.int32), axis=2) > 0
            w_vdw = jnp.where(
                any_match,
                jnp.sum(jnp.where(matches, excl_vdw_i[:, None, :], jnp.asarray(0.0, dtype=dtype)), axis=2),
                jnp.asarray(1.0, dtype=dtype),
            )
            w_elec = jnp.where(
                any_match,
                jnp.sum(jnp.where(matches, excl_elec_i[:, None, :], jnp.asarray(0.0, dtype=dtype)), axis=2),
                jnp.asarray(1.0, dtype=dtype),
            )

            sig_ij = jnp.asarray(0.5, dtype=dtype) * (sig_i[:, None] + sig_j[None, :])
            sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), jnp.asarray(1.0, dtype=dtype))
            eps_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j[None, :], jnp.asarray(0.0, dtype=dtype))) * w_vdw

            zero = jnp.asarray(0.0, dtype=dtype)
            u = dist / sig_safe
            u6 = u ** 6
            x_inv6 = jnp.asarray(1.0, dtype=dtype) / jnp.maximum(soft_alpha_lj + u6, jnp.asarray(1e-6, dtype=dtype))
            e_lj = jnp.asarray(4.0, dtype=dtype) * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)
            de_lj_over_r = (
                jnp.asarray(-24.0, dtype=dtype)
                * eps_ij
                * lam
                * (jnp.asarray(2.0, dtype=dtype) * x_inv6 - jnp.asarray(1.0, dtype=dtype))
                * (x_inv6 * x_inv6)
                * u6
                / dist_sq
            )

            dist_coul_sq = dist_sq + soft_alpha_coul
            dist_coul = jnp.sqrt(dist_coul_sq)
            qq = q_i[:, None] * q_j[None, :] * coul_const * w_elec
            if pme_alpha > 0.0:
                alpha_rc = alpha_val * dist_coul
                erfc_val = _triton_erfc(alpha_rc)
                e_coul = qq * erfc_val / dist_coul
                de_coul_over_r = (zero - qq) * (
                    erfc_val / (dist_coul * dist_coul_sq)
                    + (alpha_val * two_over_sqrt_pi * jnp.exp(zero - (alpha_rc * alpha_rc))) / dist_coul_sq
                )
            else:
                e_coul = qq / dist_coul
                de_coul_over_r = (zero - qq) / (dist_coul * dist_coul_sq)

            e_pair = jnp.where(active, e_lj + e_coul, zero)
            f_scalar = jnp.where(active, zero - (de_lj_over_r + de_coul_over_r), zero)

            acc_e = acc_e + jnp.asarray(0.5, dtype=dtype) * jnp.sum(e_pair, axis=1)
            acc_fx = acc_fx + jnp.sum(f_scalar * dx, axis=1)
            acc_fy = acc_fy + jnp.sum(f_scalar * dy, axis=1)
            acc_fz = acc_fz + jnp.sum(f_scalar * dz, axis=1)
            return (acc_e, acc_fx, acc_fy, acc_fz)

        init_carry = (
            jnp.zeros((_FLASH_BLOCK_I,), dtype=dtype),
            jnp.zeros((_FLASH_BLOCK_I,), dtype=dtype),
            jnp.zeros((_FLASH_BLOCK_I,), dtype=dtype),
            jnp.zeros((_FLASH_BLOCK_I,), dtype=dtype),
        )
        e_blk, fx_blk, fy_blk, fz_blk = jax.lax.fori_loop(0, n_blocks, _j_tile_body, init_carry)
        _pl_store(e_out_ref, (offs_i,), e_blk)
        _pl_store(fx_out_ref, (offs_i,), fx_blk)
        _pl_store(fy_out_ref, (offs_i,), fy_blk)
        _pl_store(fz_out_ref, (offs_i,), fz_blk)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(11)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [jax.ShapeDtypeStruct((N_pad,), dtype) for _ in range(4)]

    e_pad, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _flash_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_flash_explicit_nonbonded",
    )(
        pos_x,
        pos_y,
        pos_z,
        q_pad,
        sig_pad,
        eps_pad,
        mask_pad,
        excl_idx_flat,
        excl_vdw_flat,
        excl_elec_flat,
        box_vec,
    )
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1)
    return jnp.sum(e_pad[:N]), forces


def _pallas_flash_explicit_vmappable(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    soft_core_lambda: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    hb = bool(has_box)
    pa = float(pme_alpha)
    lam = float(soft_core_lambda)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, sig, eps, mask, ei, ev, ee, bx):
        return _pallas_flash_explicit_single(
            pos,
            q,
            sig,
            eps,
            mask,
            ei,
            ev,
            ee,
            bx,
            hb,
            pa,
            lam,
            interp,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, sig, eps, mask, ei, ev, ee, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                q[b_idx] if in_batched[1] else q,
                sig[b_idx] if in_batched[2] else sig,
                eps[b_idx] if in_batched[3] else eps,
                mask[b_idx] if in_batched[4] else mask,
                ei[b_idx] if in_batched[5] else ei,
                ev[b_idx] if in_batched[6] else ev,
                ee[b_idx] if in_batched[7] else ee,
                bx[b_idx] if in_batched[8] else bx,
            )

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    return _vmappable(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
    )


@functools.partial(custom_vjp, nondiff_argnums=(9, 10, 11, 12))
def _pallas_flash_explicit_diff(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    has_box: bool,
    pme_alpha: float,
    soft_core_lambda: float,
    interpret: bool,
) -> jax.Array:
    energy, _ = _pallas_flash_explicit_vmappable(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        has_box,
        pme_alpha,
        soft_core_lambda,
        interpret,
    )
    return energy


def _pallas_flash_explicit_fwd(
    positions,
    charges,
    sigmas,
    epsilons,
    atom_mask,
    excl_indices,
    excl_scales_vdw,
    excl_scales_elec,
    box,
    has_box,
    pme_alpha,
    soft_core_lambda,
    interpret,
):
    energy, forces = _pallas_flash_explicit_vmappable(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        has_box,
        pme_alpha,
        soft_core_lambda,
        interpret,
    )
    return energy, (forces, charges, sigmas, epsilons, atom_mask, excl_indices, excl_scales_vdw, excl_scales_elec, box)


def _pallas_flash_explicit_bwd(
    has_box,
    pme_alpha,
    soft_core_lambda,
    interpret,
    res,
    g,
):
    forces, charges, sigmas, epsilons, atom_mask, excl_indices, excl_scales_vdw, excl_scales_elec, box = res
    grad_pos = -g[..., None, None] * forces if g.ndim > 0 else -g * forces
    return (
        grad_pos,
        jnp.zeros_like(charges),
        jnp.zeros_like(sigmas),
        jnp.zeros_like(epsilons),
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(excl_indices),
        jnp.zeros_like(excl_scales_vdw),
        jnp.zeros_like(excl_scales_elec),
        jnp.zeros_like(box),
    )


_pallas_flash_explicit_diff.defvjp(_pallas_flash_explicit_fwd, _pallas_flash_explicit_bwd)


def pallas_flash_explicit_nonbonded(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array | None = None,
    pme_alpha: float = 0.0,
    soft_core_lambda: float = 1.0,
    interpret: bool | None = None,
) -> jax.Array:
    """2D-tiled (32, 32) FlashMD explicit nonbonded Pallas kernel with forward-force custom_vjp."""
    if interpret is None:
        interpret = not _triton_gpu_available()
    has_box = box is not None
    box_arr = (
        jnp.asarray(box, dtype=positions.dtype)
        if box is not None
        else jnp.ones((3,), dtype=positions.dtype)
    )
    if box_arr.ndim == 1 and positions.ndim == 3:
        box_arr = jnp.broadcast_to(box_arr[None, :], (positions.shape[0], 3))

    return _pallas_flash_explicit_diff(
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box_arr,
        bool(has_box),
        float(pme_alpha),
        float(soft_core_lambda),
        bool(interpret),
    )


__all__ = [
    "PallasKernelConfig",
    "gb_coulomb_energy_dense",
    "gb_energy_split_cutoff",
    "pallas_born_radii",
    "pallas_flash_explicit_nonbonded",
    "pallas_nonbonded_nl",
    "resolve_pallas_mode",
]
