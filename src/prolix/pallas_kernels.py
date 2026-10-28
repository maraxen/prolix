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
from prolix.physics.pallas_bonded import (
    _min_image_1d,
    pallas_angles_harmonic,
    pallas_bonds_2body,
    pallas_cmap_torsions,
    pallas_exceptions_14,
    pallas_pme_exclusion_pairs,
    pallas_torsions_4body,
)
from prolix.physics.pallas_implicit import (
    pallas_gb_ace_nl,
    pallas_sasa_nl,
)


def _normalize_box(box: jax.Array | None, positions: jax.Array) -> tuple[jax.Array, bool]:
    """Normalize optional box (None, (3,), or lower-triangular (3, 3)) to (6,) or (B, 6) array [ax, by, cz, bx, cx, cy]."""
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
        elif b_raw.shape[-1] == 6:
            b = b_raw
        else:
            z = jnp.zeros_like(b_raw[..., 0])
            b = jnp.stack([b_raw[..., 0], b_raw[..., 1], b_raw[..., 2], z, z, z], axis=-1)
        has_box = True
    if b.ndim == 1 and positions.ndim == 3:
        b = jnp.broadcast_to(b[None, :], (positions.shape[0], 6))
    return b, has_box


def _min_image_triclinic_xyz(
    dx: jax.Array,
    dy: jax.Array,
    dz: jax.Array,
    ax: jax.Array,
    by: jax.Array,
    cz: jax.Array,
    bx: jax.Array,
    cx: jax.Array,
    cy: jax.Array,
    has_box: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """3-step sequential reduction (z -> y -> x) for orthogonal or lower-triangular triclinic box."""
    if not has_box:
        return dx, dy, dz
    dtype = dx.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    cz_pos = cz > zero
    cz_safe = jnp.where(cz_pos, cz, one)
    sz = jnp.where(cz_pos, jnp.floor(dz / cz_safe + half), zero)
    dx = dx - sz * cx
    dy = dy - sz * cy
    dz = dz - sz * cz

    by_pos = by > zero
    by_safe = jnp.where(by_pos, by, one)
    sy = jnp.where(by_pos, jnp.floor(dy / by_safe + half), zero)
    dx = dx - sy * bx
    dy = dy - sy * by

    ax_pos = ax > zero
    ax_safe = jnp.where(ax_pos, ax, one)
    sx = jnp.where(ax_pos, jnp.floor(dx / ax_safe + half), zero)
    dx = dx - sx * ax
    return dx, dy, dz


def _min_image_triclinic_vec(dr: jax.Array, box: jax.Array, has_box: bool) -> jax.Array:
    """Apply 3-step triclinic minimum-image reduction to (..., 3) displacement array."""
    if not has_box:
        return dr
    dx, dy, dz = _min_image_triclinic_xyz(
        dr[..., 0],
        dr[..., 1],
        dr[..., 2],
        box[0],
        box[1],
        box[2],
        box[3],
        box[4],
        box[5],
        has_box,
    )
    return jnp.stack([dx, dy, dz], axis=-1)

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
    del interpret
    positions, radii, scaled_radii, neighbor_idx = res

    if positions.ndim == 3:
        B = positions.shape[0]
        b_rad = radii.ndim > 1
        b_sc = scaled_radii.ndim > 1
        b_nbr = neighbor_idx.ndim > 2

        def _ref_batch(pos, rad, sc_rad):
            return jax.vmap(
                lambda b: compute_born_radii_neighbor_list(
                    pos[b],
                    rad[b] if b_rad else rad,
                    neighbor_idx[b] if b_nbr else neighbor_idx,
                    dielectric_offset=dielectric_offset,
                    scaled_radii=sc_rad[b] if b_sc else sc_rad,
                )
            )(jnp.arange(B))

        _, vjp_fn = jax.vjp(_ref_batch, positions, radii, scaled_radii)
        grad_pos, grad_rad, grad_sc_rad = vjp_fn(g)
        return (grad_pos, grad_rad, grad_sc_rad, jnp.zeros_like(neighbor_idx))

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
        if positions.ndim == 3:
            B = positions.shape[0]
            return jax.vmap(
                lambda b: compute_born_radii_neighbor_list(
                    positions[b],
                    radii[b] if radii.ndim > 1 else radii,
                    neighbor_idx[b] if neighbor_idx.ndim > 2 else neighbor_idx,
                    dielectric_offset=dielectric_offset,
                    scaled_radii=scaled_radii[b] if scaled_radii.ndim > 1 else scaled_radii,
                )
            )(jnp.arange(B))
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """Single-system vectorized (32, 32) Pallas O(N*K) nonbonded energy & forces."""
    N, K = neighbor_idx.shape
    M = excl_indices.shape[1]
    if M > _NB_MAX_EXCL:
        raise ValueError(f"excl_indices width {M} exceeds _NB_MAX_EXCL={_NB_MAX_EXCL}")
    dtype = positions.dtype

    n_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    N_pad = n_blocks * _NB_BLOCK_N
    pad_n = N_pad - N

    k_blocks = max(1, (K + _NB_BLOCK_K - 1) // _NB_BLOCK_K)
    K_pad = k_blocks * _NB_BLOCK_K
    pad_k = K_pad - K

    pad_m = max(0, _NB_MAX_EXCL - M)
    pad_n_excl = N_pad - excl_indices.shape[0]
    pad_n_nbr = N_pad - neighbor_idx.shape[0]

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
    if pad_n_excl > 0 or pad_m > 0:
        excl_idx_pad = jnp.pad(excl_idx_pad, ((0, pad_n_excl), (0, pad_m)), constant_values=-1)
        excl_vdw_pad = jnp.pad(excl_vdw_pad, ((0, pad_n_excl), (0, pad_m)), constant_values=1.0)
        excl_elec_pad = jnp.pad(excl_elec_pad, ((0, pad_n_excl), (0, pad_m)), constant_values=1.0)

    nbr_pad = neighbor_idx.astype(jnp.int32)
    if pad_n_nbr > 0 or pad_k > 0:
        nbr_pad = jnp.pad(nbr_pad, ((0, pad_n_nbr), (0, pad_k)), constant_values=N)

    pos_x = jnp.asarray(pos_pad[:, 0])
    pos_y = jnp.asarray(pos_pad[:, 1])
    pos_z = jnp.asarray(pos_pad[:, 2])
    nbr_flat = jnp.asarray(nbr_pad.reshape(N_pad * K_pad))
    excl_idx_flat = jnp.asarray(excl_idx_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_vdw_flat = jnp.asarray(excl_vdw_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_elec_flat = jnp.asarray(excl_elec_pad.reshape(N_pad * _NB_MAX_EXCL))
    box_vec = jnp.asarray(box.astype(dtype))
    lam_vec = jnp.asarray(lam_arr.reshape((1,)).astype(dtype))

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
        lam_ref,
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
        valid_excl_i = (excl_idx_i >= 0) & (excl_idx_i < N)

        ax = _pl_load(box_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (jnp.ones((1,), dtype=jnp.int32),))[0]
        cz = _pl_load(box_ref, (jnp.full((1,), 2, dtype=jnp.int32),))[0]
        bx_t = _pl_load(box_ref, (jnp.full((1,), 3, dtype=jnp.int32),))[0]
        cx_t = _pl_load(box_ref, (jnp.full((1,), 4, dtype=jnp.int32),))[0]
        cy_t = _pl_load(box_ref, (jnp.full((1,), 5, dtype=jnp.int32),))[0]

        one_sc = jnp.asarray(1.0, dtype=dtype)
        lam = _pl_load(lam_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        one_minus_lam = one_sc - lam
        soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (
            one_minus_lam ** int(softcore_exponent)
        )
        coul_const = jnp.asarray(coulomb_constant, dtype=dtype)
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

            dx, dy, dz = _min_image_triclinic_xyz(
                px_i[:, None] - px_j,
                py_i[:, None] - py_j,
                pz_i[:, None] - pz_j,
                ax,
                by,
                cz,
                bx_t,
                cx_t,
                cy_t,
                has_box,
            )

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
                    erf_v = erf(alpha_r) if (interpret and dtype == jnp.float64) else _triton_erf(alpha_r)
                    eff_w = w_elec - erf_v
                    e_coul = qq * eff_w / dist
                    de_coul_dr = qq * (
                        (zero - eff_w) / r_sq
                        - (alpha_val * two_over_sqrt_pi * exp_term) / dist
                    )
                else:
                    erfc_val = erfc(alpha_r) if (interpret and dtype == jnp.float64) else _triton_erfc(alpha_r)
                    e_coul = qq * w_elec * erfc_val / dist
                    de_coul_dr = qq * w_elec * (
                        (zero - erfc_val) / r_sq
                        - (alpha_val * two_over_sqrt_pi * exp_term) / dist
                    )
            else:
                valid_coul = dist >= jnp.asarray(1e-4, dtype=dtype)
                e_coul = jnp.where(valid_coul, qq * w_elec / dist, zero)
                de_coul_dr = jnp.where(valid_coul, (zero - qq) * w_elec / r_sq, zero)

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

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(13)]
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
        lam_vec,
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    hb = bool(has_box)
    pa = float(pme_alpha)
    co = float(cutoff)
    cc = float(coulomb_constant)
    sw = float(lj_switch_width)
    sca = float(softcore_alpha)
    sce = int(softcore_exponent)
    inc_erf = bool(include_pme_excl_erf)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, sig, eps, mask, nbr, ei, ev, ee, bx, lam):
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
            lam,
            hb,
            pa,
            co,
            cc,
            sw,
            sca,
            sce,
            inc_erf,
            interp,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, sig, eps, mask, nbr, ei, ev, ee, bx, lam):
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
                lam[b_idx] if in_batched[10] else lam,
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
        lam_arr,
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Chunked O(chunk_size*K) analytical parameter gradients (grad_q, grad_sig, grad_eps, grad_excl_vdw, grad_excl_elec, grad_lam)."""
    N, K = neighbor_idx.shape
    M_excl = excl_indices.shape[1]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)

    if N == 0 or K == 0:
        return (
            jnp.zeros_like(charges),
            jnp.zeros_like(sigmas),
            jnp.zeros_like(epsilons),
            jnp.zeros_like(excl_scales_vdw),
            jnp.zeros_like(excl_scales_elec),
            jnp.zeros_like(lam_arr),
        )

    chunk_size = min(max(((N + 31) // 32) * 32, 32), 256)
    n_chunks = (N + chunk_size - 1) // chunk_size
    N_pad = n_chunks * chunk_size
    pad_n = N_pad - N
    pad_n_excl = N_pad - excl_indices.shape[0]
    pad_n_nbr = N_pad - neighbor_idx.shape[0]

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    q_pad = jnp.pad(charges.astype(dtype), (0, pad_n)) if pad_n > 0 else charges.astype(dtype)
    sig_pad = jnp.pad(sigmas.astype(dtype), (0, pad_n), constant_values=1.0) if pad_n > 0 else sigmas.astype(dtype)
    eps_pad = jnp.pad(epsilons.astype(dtype), (0, pad_n)) if pad_n > 0 else epsilons.astype(dtype)
    mask_pad = jnp.pad(jnp.asarray(atom_mask, dtype=jnp.bool_), (0, pad_n), constant_values=False) if pad_n > 0 else jnp.asarray(atom_mask, dtype=jnp.bool_)
    nbr_pad = jnp.pad(neighbor_idx.astype(jnp.int32), ((0, pad_n_nbr), (0, 0)), constant_values=N) if pad_n_nbr > 0 else neighbor_idx.astype(jnp.int32)
    ei_pad = jnp.pad(excl_indices.astype(jnp.int32), ((0, pad_n_excl), (0, 0)), constant_values=-1) if pad_n_excl > 0 else excl_indices.astype(jnp.int32)
    ev_pad = jnp.pad(excl_scales_vdw.astype(dtype), ((0, pad_n_excl), (0, 0)), constant_values=1.0) if pad_n_excl > 0 else excl_scales_vdw.astype(dtype)
    ee_pad = jnp.pad(excl_scales_elec.astype(dtype), ((0, pad_n_excl), (0, 0)), constant_values=1.0) if pad_n_excl > 0 else excl_scales_elec.astype(dtype)

    lam = lam_arr.reshape(())
    exp_f = int(softcore_exponent)
    one_minus_lam = one - lam
    soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (one_minus_lam ** exp_f)
    if exp_f >= 1:
        d_denom_dlam = -jnp.asarray(float(exp_f) * softcore_alpha, dtype=dtype) * (
            one_minus_lam ** (exp_f - 1)
        )
    else:
        d_denom_dlam = zero
    coul_const = jnp.asarray(coulomb_constant, dtype=dtype)
    alpha_val = jnp.asarray(pme_alpha, dtype=dtype)

    def _chunk_fn(carry_lam, c_idx):
        i_start = jnp.int32(c_idx * chunk_size)
        z_i32 = jnp.int32(0)
        i_offs = i_start + jnp.arange(chunk_size, dtype=jnp.int32)
        pos_i = jax.lax.dynamic_slice(pos_pad, (i_start, z_i32), (chunk_size, 3))
        q_i = jax.lax.dynamic_slice(q_pad, (i_start,), (chunk_size,))
        sig_i = jax.lax.dynamic_slice(sig_pad, (i_start,), (chunk_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (i_start,), (chunk_size,))
        mask_i = jax.lax.dynamic_slice(mask_pad, (i_start,), (chunk_size,)) & (i_offs < N)
        nbr_i = jax.lax.dynamic_slice(nbr_pad, (i_start, z_i32), (chunk_size, K))
        ei_i = jax.lax.dynamic_slice(ei_pad, (i_start, z_i32), (chunk_size, M_excl))
        ev_i = jax.lax.dynamic_slice(ev_pad, (i_start, z_i32), (chunk_size, M_excl))
        ee_i = jax.lax.dynamic_slice(ee_pad, (i_start, z_i32), (chunk_size, M_excl))

        safe_j = jnp.clip(nbr_i, 0, max(N - 1, 0))
        dr = _min_image_triclinic_vec(pos_i[:, None, :] - pos_pad[safe_j], box, has_box)

        r_sq = jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=dtype)
        dist = jnp.sqrt(r_sq)
        active = (
            mask_i[:, None]
            & mask_pad[safe_j]
            & (nbr_i >= 0)
            & (nbr_i < N)
            & (nbr_i != i_offs[:, None])
            & (dist < jnp.asarray(cutoff, dtype=dtype))
        )

        if M_excl > 0:
            matches = (nbr_i[:, :, None] == ei_i[:, None, :]) & (ei_i[:, None, :] >= 0)
            any_match = jnp.any(matches, axis=2)
            w_vdw = jnp.where(any_match, jnp.sum(jnp.where(matches, ev_i[:, None, :], zero), axis=2), one)
            w_elec = jnp.where(any_match, jnp.sum(jnp.where(matches, ee_i[:, None, :], zero), axis=2), one)
        else:
            matches = jnp.zeros((chunk_size, K, 0), dtype=jnp.bool_)
            w_vdw = jnp.ones((chunk_size, K), dtype=dtype)
            w_elec = jnp.ones((chunk_size, K), dtype=dtype)

        sig_ij = half * (sig_i[:, None] + sig_pad[safe_j])
        sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
        eps_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_pad[safe_j], zero) + jnp.asarray(1e-30, dtype=dtype))

        u = dist / sig_safe
        u6 = u ** 6
        denom_lj = jnp.maximum(soft_alpha_lj + u6, jnp.asarray(1e-6, dtype=dtype))
        x_inv6 = one / denom_lj
        s_w, _ = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)

        v_unsw = jnp.asarray(4.0, dtype=dtype) * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)
        e_lj_unweighted = v_unsw * s_w

        de_ds_ij = (
            jnp.asarray(24.0, dtype=dtype)
            * eps_ij
            * lam
            * (jnp.asarray(2.0, dtype=dtype) * x_inv6 - one)
            * (x_inv6 * x_inv6)
            * u6
            / sig_safe
        )
        grad_sig_i = jnp.sum(jnp.where(active, half * w_vdw * s_w * de_ds_ij, zero), axis=1)

        v_shape = jnp.asarray(4.0, dtype=dtype) * lam * (x_inv6 * x_inv6 - x_inv6)
        deps_ij_depsi = half * eps_pad[safe_j] / jnp.maximum(eps_ij, jnp.asarray(1e-15, dtype=dtype))
        grad_eps_i = jnp.sum(jnp.where(active, w_vdw * s_w * v_shape * deps_ij_depsi, zero), axis=1)

        # dE_LJ / d(lam)
        dv_dlam = jnp.asarray(4.0, dtype=dtype) * eps_ij * (
            (x_inv6 * x_inv6 - x_inv6)
            + lam * (jnp.asarray(2.0, dtype=dtype) * x_inv6 - one) * (x_inv6 * x_inv6) * (-d_denom_dlam)
        )
        dlam_chunk = jnp.sum(jnp.where(active, half * w_vdw * s_w * dv_dlam, zero))

        q_j = q_pad[safe_j]
        if pme_alpha > 0.0:
            if include_pme_excl_erf:
                coul_kernel = (w_elec - erf(alpha_val * dist)) / dist
                dcoul_dw_elec = one / dist
            else:
                erfc_v = erfc(alpha_val * dist)
                coul_kernel = (w_elec * erfc_v) / dist
                dcoul_dw_elec = erfc_v / dist
        else:
            valid_coul = dist >= jnp.asarray(1e-4, dtype=dtype)
            coul_kernel = jnp.where(valid_coul, w_elec / dist, zero)
            dcoul_dw_elec = jnp.where(valid_coul, one / dist, zero)

        grad_q_i = jnp.sum(jnp.where(active, coul_const * q_j * coul_kernel, zero), axis=1)

        if M_excl > 0:
            de_dw_vdw = jnp.where(active, half * e_lj_unweighted, zero)
            de_dw_elec = jnp.where(active, half * coul_const * q_i[:, None] * q_j * dcoul_dw_elec, zero)
            grad_ev_i = jnp.sum(jnp.where(matches, de_dw_vdw[:, :, None], zero), axis=1)
            grad_ee_i = jnp.sum(jnp.where(matches, de_dw_elec[:, :, None], zero), axis=1)
        else:
            grad_ev_i = jnp.zeros((chunk_size, 0), dtype=dtype)
            grad_ee_i = jnp.zeros((chunk_size, 0), dtype=dtype)

        return carry_lam + dlam_chunk, (grad_q_i, grad_sig_i, grad_eps_i, grad_ev_i, grad_ee_i)

    grad_lam_tot, (gq_c, gs_c, ge_c, gev_c, gee_c) = jax.lax.scan(
        _chunk_fn, zero, jnp.arange(n_chunks, dtype=jnp.int32)
    )
    n_excl_orig = excl_indices.shape[0]
    return (
        gq_c.reshape(N_pad)[:N].astype(charges.dtype),
        gs_c.reshape(N_pad)[:N].astype(sigmas.dtype),
        ge_c.reshape(N_pad)[:N].astype(epsilons.dtype),
        gev_c.reshape(N_pad, M_excl)[:n_excl_orig].astype(excl_scales_vdw.dtype),
        gee_c.reshape(N_pad, M_excl)[:n_excl_orig].astype(excl_scales_elec.dtype),
        grad_lam_tot.reshape(lam_arr.shape).astype(lam_arr.dtype),
    )


@functools.partial(custom_vjp, nondiff_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19))
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
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
        lam_arr,
        has_box,
        pme_alpha,
        cutoff,
        coulomb_constant,
        lj_switch_width,
        softcore_alpha,
        softcore_exponent,
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
    lam_arr,
    has_box,
    pme_alpha,
    cutoff,
    coulomb_constant,
    lj_switch_width,
    softcore_alpha,
    softcore_exponent,
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
        lam_arr,
        has_box,
        pme_alpha,
        cutoff,
        coulomb_constant,
        lj_switch_width,
        softcore_alpha,
        softcore_exponent,
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
        lam_arr,
    )
    return energy, res


def _pallas_nonbonded_nl_bwd(
    has_box,
    pme_alpha,
    cutoff,
    coulomb_constant,
    lj_switch_width,
    softcore_alpha,
    softcore_exponent,
    include_pme_excl_erf,
    interpret,
    res,
    g,
):
    del interpret
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
        lam_arr,
    ) = res
    grad_pos = -g[..., None, None] * forces if g.ndim > 0 else -g * forces

    if positions.ndim == 2:
        gq, gs, ge, gev, gee, glam = _nonbonded_nl_param_grads(
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
            lam_arr,
            has_box,
            pme_alpha,
            cutoff,
            coulomb_constant,
            lj_switch_width,
            softcore_alpha,
            softcore_exponent,
            include_pme_excl_erf,
        )
        grad_q = (g * gq).astype(charges.dtype)
        grad_sig = (g * gs).astype(sigmas.dtype)
        grad_eps = (g * ge).astype(epsilons.dtype)
        grad_ev = (g * gev).astype(excl_scales_vdw.dtype)
        grad_ee = (g * gee).astype(excl_scales_elec.dtype)
        grad_lam = (g * glam).astype(lam_arr.dtype)
    else:
        B = positions.shape[0]
        b_q = charges.ndim > 1
        b_sig = sigmas.ndim > 1
        b_eps = epsilons.ndim > 1
        b_msk = atom_mask.ndim > 1
        b_nbr = neighbor_idx.ndim > 2
        b_ei = excl_indices.ndim > 2
        b_ev = excl_scales_vdw.ndim > 2
        b_ee = excl_scales_elec.ndim > 2
        b_box = box.ndim > 1
        b_lam = lam_arr.ndim > 1

        def _step(b):
            gq_b, gs_b, ge_b, gev_b, gee_b, glam_b = _nonbonded_nl_param_grads(
                positions[b],
                charges[b] if b_q else charges,
                sigmas[b] if b_sig else sigmas,
                epsilons[b] if b_eps else epsilons,
                atom_mask[b] if b_msk else atom_mask,
                neighbor_idx[b] if b_nbr else neighbor_idx,
                excl_indices[b] if b_ei else excl_indices,
                excl_scales_vdw[b] if b_ev else excl_scales_vdw,
                excl_scales_elec[b] if b_ee else excl_scales_elec,
                box[b] if b_box else box,
                lam_arr[b] if b_lam else lam_arr,
                has_box,
                pme_alpha,
                cutoff,
                coulomb_constant,
                lj_switch_width,
                softcore_alpha,
                softcore_exponent,
                include_pme_excl_erf,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (
                (g_b * gq_b).astype(charges.dtype),
                (g_b * gs_b).astype(sigmas.dtype),
                (g_b * ge_b).astype(epsilons.dtype),
                (g_b * gev_b).astype(excl_scales_vdw.dtype),
                (g_b * gee_b).astype(excl_scales_elec.dtype),
                (g_b * glam_b).astype(lam_arr.dtype),
            )

        gq_all, gs_all, ge_all, gev_all, gee_all, glam_all = jax.lax.map(_step, jnp.arange(B))
        grad_q = gq_all if b_q else jnp.sum(gq_all, axis=0)
        grad_sig = gs_all if b_sig else jnp.sum(gs_all, axis=0)
        grad_eps = ge_all if b_eps else jnp.sum(ge_all, axis=0)
        grad_ev = gev_all if b_ev else jnp.sum(gev_all, axis=0)
        grad_ee = gee_all if b_ee else jnp.sum(gee_all, axis=0)
        grad_lam = glam_all if b_lam else jnp.sum(glam_all, axis=0)

    return (
        grad_pos,
        grad_q,
        grad_sig,
        grad_eps,
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(neighbor_idx),
        jnp.zeros_like(excl_indices),
        grad_ev,
        grad_ee,
        jnp.zeros_like(box),
        grad_lam,
    )


_pallas_nonbonded_nl_diff.defvjp(_pallas_nonbonded_nl_fwd, _pallas_nonbonded_nl_bwd)


def _export_nonbonded_nl_one(
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
    lam: jax.Array,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
) -> jax.Array:
    """Portable XLA fallback for pallas_nonbonded_nl under export_context()."""
    N, K = neighbor_idx.shape
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    if N == 0 or K == 0:
        return zero

    safe_j = jnp.clip(neighbor_idx, 0, max(N - 1, 0))
    dr = _min_image_triclinic_vec(positions[:, None, :] - positions[safe_j], box, has_box)

    dist = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=dtype))
    m_bool = jnp.asarray(atom_mask, dtype=jnp.bool_)
    active = (
        m_bool[:, None]
        & m_bool[safe_j]
        & (neighbor_idx >= 0)
        & (neighbor_idx < N)
        & (neighbor_idx != jnp.arange(N, dtype=jnp.int32)[:, None])
        & (dist < jnp.asarray(cutoff, dtype=dtype))
    )
    w_vdw, w_elec = get_neighbor_exclusion_scales(
        excl_indices, excl_scales_vdw, excl_scales_elec, neighbor_idx
    )
    w_vdw = w_vdw.astype(dtype)
    w_elec = w_elec.astype(dtype)

    sig_ij = half * (sigmas[:, None] + sigmas[safe_j])
    sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
    eps_ij = jnp.sqrt(jnp.maximum(epsilons[:, None] * epsilons[safe_j], zero) + jnp.asarray(1e-30, dtype=dtype))
    lam_val = lam.reshape(())
    one_minus_lam = one - lam_val
    soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (
        one_minus_lam ** int(softcore_exponent)
    )
    u6 = (dist / sig_safe) ** 6
    x_inv6 = one / jnp.maximum(soft_alpha_lj + u6, jnp.asarray(1e-6, dtype=dtype))
    s_w, _ = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
    e_lj = w_vdw * jnp.asarray(4.0, dtype=dtype) * eps_ij * lam_val * (x_inv6 * x_inv6 - x_inv6) * s_w

    qq = jnp.asarray(coulomb_constant, dtype=dtype) * charges[:, None] * charges[safe_j]
    if pme_alpha > 0.0:
        alpha_r = jnp.asarray(pme_alpha, dtype=dtype) * dist
        if include_pme_excl_erf:
            e_coul = qq * (w_elec - erf(alpha_r)) / dist
        else:
            e_coul = qq * w_elec * erfc(alpha_r) / dist
    else:
        e_coul = jnp.where(dist >= jnp.asarray(1e-4, dtype=dtype), qq * w_elec / dist, zero)

    return half * jnp.sum(jnp.where(active, e_lj + e_coul, zero))


def pallas_nonbonded_nl(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array | None,
    neighbor_idx: jax.Array | None = None,
    excl_indices: jax.Array | None = None,
    excl_scales_vdw: jax.Array | None = None,
    excl_scales_elec: jax.Array | None = None,
    box: jax.Array | None = None,
    pme_alpha: float = 0.0,
    cutoff: float = 10.0,
    coulomb_constant: float = constants.COULOMB_CONSTANT,
    lj_switch_width: float = 0.0,
    soft_core_lambda: float | jax.Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    include_pme_excl_erf: bool = True,
    interpret: bool | None = None,
) -> jax.Array:
    """Fused O(N*K) neighbor-list LJ + Coulomb/PME direct-space Pallas kernel."""
    if (
        excl_scales_elec is None
        and atom_mask is not None
        and jnp.issubdtype(jnp.asarray(atom_mask).dtype, jnp.integer)
        and jnp.asarray(atom_mask).ndim >= 2
        and (
            neighbor_idx is None
            or jnp.issubdtype(jnp.asarray(neighbor_idx).dtype, jnp.integer)
        )
        and (
            excl_indices is None
            or jnp.issubdtype(jnp.asarray(excl_indices).dtype, jnp.floating)
        )
    ):
        # Called with 8-arg signature: (pos, q, sig, eps, neighbor_idx, excl_indices, excl_vdw, excl_elec)
        excl_scales_elec = excl_scales_vdw
        excl_scales_vdw = excl_indices
        excl_indices = neighbor_idx
        neighbor_idx = atom_mask
        atom_mask = None

    dtype = positions.dtype
    N = positions.shape[-2]
    msk_arr = (
        jnp.ones(positions.shape[:-1], dtype=jnp.bool_)
        if atom_mask is None
        else jnp.asarray(atom_mask)
    )
    if excl_indices is None:
        ei_arr = jnp.zeros((N, 0), dtype=jnp.int32)
        ev_arr = jnp.zeros((N, 0), dtype=dtype)
        ee_arr = jnp.zeros((N, 0), dtype=dtype)
    else:
        ei_arr = jnp.asarray(excl_indices, dtype=jnp.int32)
        ev_arr = (
            jnp.zeros_like(ei_arr, dtype=dtype)
            if excl_scales_vdw is None
            else jnp.asarray(excl_scales_vdw, dtype=dtype)
        )
        ee_arr = (
            jnp.zeros_like(ei_arr, dtype=dtype)
            if excl_scales_elec is None
            else jnp.asarray(excl_scales_elec, dtype=dtype)
        )

    box_arr, has_box = _normalize_box(box, positions)
    lam_raw = jnp.asarray(soft_core_lambda, dtype=dtype)
    lam_arr = lam_raw.reshape((1,)) if lam_raw.ndim == 0 else lam_raw.reshape((-1, 1))

    if in_export_context():
        if positions.ndim == 3:
            return jax.vmap(
                lambda b: _export_nonbonded_nl_one(
                    positions[b],
                    charges[b] if charges.ndim == 2 else charges,
                    sigmas[b] if sigmas.ndim == 2 else sigmas,
                    epsilons[b] if epsilons.ndim == 2 else epsilons,
                    msk_arr[b] if msk_arr.ndim == 2 else msk_arr,
                    neighbor_idx[b] if neighbor_idx.ndim == 3 else neighbor_idx,
                    ei_arr[b] if ei_arr.ndim == 3 else ei_arr,
                    ev_arr[b] if ev_arr.ndim == 3 else ev_arr,
                    ee_arr[b] if ee_arr.ndim == 3 else ee_arr,
                    box_arr[b],
                    lam_arr[b] if lam_arr.ndim == 2 else lam_arr,
                    has_box,
                    float(pme_alpha),
                    float(cutoff),
                    float(coulomb_constant),
                    float(lj_switch_width),
                    float(softcore_alpha),
                    int(softcore_exponent),
                    bool(include_pme_excl_erf),
                )
            )(jnp.arange(positions.shape[0]))
        return _export_nonbonded_nl_one(
            positions,
            charges,
            sigmas,
            epsilons,
            msk_arr,
            neighbor_idx,
            ei_arr,
            ev_arr,
            ee_arr,
            box_arr,
            lam_arr,
            has_box,
            float(pme_alpha),
            float(cutoff),
            float(coulomb_constant),
            float(lj_switch_width),
            float(softcore_alpha),
            int(softcore_exponent),
            bool(include_pme_excl_erf),
        )

    if interpret is None:
        interpret = not _triton_gpu_available()

    return _pallas_nonbonded_nl_diff(
        positions,
        jnp.asarray(charges, dtype=dtype),
        jnp.asarray(sigmas, dtype=dtype),
        jnp.asarray(epsilons, dtype=dtype),
        msk_arr,
        jnp.asarray(neighbor_idx, dtype=jnp.int32),
        ei_arr,
        ev_arr,
        ee_arr,
        box_arr,
        lam_arr,
        bool(has_box),
        float(pme_alpha),
        float(cutoff),
        float(coulomb_constant),
        float(lj_switch_width),
        float(softcore_alpha),
        int(softcore_exponent),
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    softcore_alpha: float,
    softcore_exponent: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """Single-system 2D-tiled (32, 32) Pallas dense explicit nonbonded energy & forces."""
    N = positions.shape[0]
    M = excl_indices.shape[1]
    if M > _NB_MAX_EXCL:
        raise ValueError(f"excl_indices width {M} exceeds _NB_MAX_EXCL={_NB_MAX_EXCL}")
    dtype = positions.dtype

    n_blocks = max(1, (N + _FLASH_BLOCK_I - 1) // _FLASH_BLOCK_I)
    N_pad = n_blocks * _FLASH_BLOCK_I
    pad_n = N_pad - N
    pad_m = max(0, _NB_MAX_EXCL - M)
    pad_n_excl = N_pad - excl_indices.shape[0]

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
    if pad_n_excl > 0 or pad_m > 0:
        excl_idx_pad = jnp.pad(excl_idx_pad, ((0, pad_n_excl), (0, pad_m)), constant_values=-1)
        excl_vdw_pad = jnp.pad(excl_vdw_pad, ((0, pad_n_excl), (0, pad_m)), constant_values=1.0)
        excl_elec_pad = jnp.pad(excl_elec_pad, ((0, pad_n_excl), (0, pad_m)), constant_values=1.0)

    pos_x = jnp.asarray(pos_pad[:, 0])
    pos_y = jnp.asarray(pos_pad[:, 1])
    pos_z = jnp.asarray(pos_pad[:, 2])
    excl_idx_flat = jnp.asarray(excl_idx_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_vdw_flat = jnp.asarray(excl_vdw_pad.reshape(N_pad * _NB_MAX_EXCL))
    excl_elec_flat = jnp.asarray(excl_elec_pad.reshape(N_pad * _NB_MAX_EXCL))
    box_vec = jnp.asarray(box.astype(dtype))
    lam_vec = jnp.asarray(lam_arr.reshape((1,)).astype(dtype))

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
        lam_ref,
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
        valid_excl_i = (excl_idx_i >= 0) & (excl_idx_i < N)

        ax = _pl_load(box_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (jnp.ones((1,), dtype=jnp.int32),))[0]
        cz = _pl_load(box_ref, (jnp.full((1,), 2, dtype=jnp.int32),))[0]
        bx_t = _pl_load(box_ref, (jnp.full((1,), 3, dtype=jnp.int32),))[0]
        cx_t = _pl_load(box_ref, (jnp.full((1,), 4, dtype=jnp.int32),))[0]
        cy_t = _pl_load(box_ref, (jnp.full((1,), 5, dtype=jnp.int32),))[0]

        lam = _pl_load(lam_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        one_minus_lam = jnp.asarray(1.0, dtype=dtype) - lam
        soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (
            one_minus_lam ** int(softcore_exponent)
        )
        soft_alpha_coul = jnp.asarray(2.0, dtype=dtype) * one_minus_lam
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

            dx, dy, dz = _min_image_triclinic_xyz(
                px_i[:, None] - px_j[None, :],
                py_i[:, None] - py_j[None, :],
                pz_i[:, None] - pz_j[None, :],
                ax,
                by,
                cz,
                bx_t,
                cx_t,
                cy_t,
                has_box,
            )

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
                erfc_val = erfc(alpha_rc) if (interpret and dtype == jnp.float64) else _triton_erfc(alpha_rc)
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

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(12)]
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
        lam_vec,
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    softcore_alpha: float,
    softcore_exponent: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    hb = bool(has_box)
    pa = float(pme_alpha)
    sca = float(softcore_alpha)
    sce = int(softcore_exponent)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, sig, eps, mask, ei, ev, ee, bx, lam):
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
            lam,
            hb,
            pa,
            sca,
            sce,
            interp,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, sig, eps, mask, ei, ev, ee, bx, lam):
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
                lam[b_idx] if in_batched[9] else lam,
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
        lam_arr,
    )


def _export_flash_one(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    softcore_alpha: float,
    softcore_exponent: int,
) -> jax.Array:
    """Chunked O(N * chunk_size) portable XLA reference for pallas_flash_explicit_nonbonded."""
    N = positions.shape[0]
    M_excl = excl_indices.shape[1]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    if N == 0:
        return zero

    chunk_size = min(max(((N + 31) // 32) * 32, 32), 256)
    n_chunks = (N + chunk_size - 1) // chunk_size
    N_pad = n_chunks * chunk_size
    pad_n = N_pad - N
    pad_n_excl = N_pad - excl_indices.shape[0]

    pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0))) if pad_n > 0 else positions
    q_pad = jnp.pad(charges.astype(dtype), (0, pad_n)) if pad_n > 0 else charges.astype(dtype)
    sig_pad = jnp.pad(sigmas.astype(dtype), (0, pad_n), constant_values=1.0) if pad_n > 0 else sigmas.astype(dtype)
    eps_pad = jnp.pad(epsilons.astype(dtype), (0, pad_n)) if pad_n > 0 else epsilons.astype(dtype)
    mask_pad = jnp.pad(jnp.asarray(atom_mask, dtype=jnp.bool_), (0, pad_n), constant_values=False) if pad_n > 0 else jnp.asarray(atom_mask, dtype=jnp.bool_)
    ei_pad = jnp.pad(excl_indices.astype(jnp.int32), ((0, pad_n_excl), (0, 0)), constant_values=-1) if pad_n_excl > 0 else excl_indices.astype(jnp.int32)
    ev_pad = jnp.pad(excl_scales_vdw.astype(dtype), ((0, pad_n_excl), (0, 0)), constant_values=1.0) if pad_n_excl > 0 else excl_scales_vdw.astype(dtype)
    ee_pad = jnp.pad(excl_scales_elec.astype(dtype), ((0, pad_n_excl), (0, 0)), constant_values=1.0) if pad_n_excl > 0 else excl_scales_elec.astype(dtype)

    lam = lam_arr.reshape(())
    one_minus_lam = one - lam
    soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (
        one_minus_lam ** int(softcore_exponent)
    )
    soft_alpha_coul = jnp.asarray(2.0, dtype=dtype) * one_minus_lam
    coul_const = jnp.asarray(constants.COULOMB_CONSTANT, dtype=dtype)
    alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
    j_offs = jnp.arange(N, dtype=jnp.int32)
    pos_j = pos_pad[:N]
    q_j = q_pad[:N]
    sig_j = sig_pad[:N]
    eps_j = eps_pad[:N]
    mask_j = mask_pad[:N]

    def _chunk_fn(acc_e, c_idx):
        i_start = jnp.int32(c_idx * chunk_size)
        z_i32 = jnp.int32(0)
        i_offs = i_start + jnp.arange(chunk_size, dtype=jnp.int32)
        pos_i = jax.lax.dynamic_slice(pos_pad, (i_start, z_i32), (chunk_size, 3))
        q_i = jax.lax.dynamic_slice(q_pad, (i_start,), (chunk_size,))
        sig_i = jax.lax.dynamic_slice(sig_pad, (i_start,), (chunk_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (i_start,), (chunk_size,))
        mask_i = jax.lax.dynamic_slice(mask_pad, (i_start,), (chunk_size,)) & (i_offs < N)
        ei_i = jax.lax.dynamic_slice(ei_pad, (i_start, z_i32), (chunk_size, M_excl))
        ev_i = jax.lax.dynamic_slice(ev_pad, (i_start, z_i32), (chunk_size, M_excl))
        ee_i = jax.lax.dynamic_slice(ee_pad, (i_start, z_i32), (chunk_size, M_excl))

        active = mask_i[:, None] & mask_j[None, :] & (i_offs[:, None] != j_offs[None, :])
        dr = _min_image_triclinic_vec(pos_i[:, None, :] - pos_j[None, :, :], box, has_box)
        dist_sq = jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-10, dtype=dtype)
        dist = jnp.sqrt(dist_sq)

        if M_excl > 0:
            matches = (j_offs[None, :, None] == ei_i[:, None, :]) & (ei_i[:, None, :] >= 0)
            any_match = jnp.any(matches, axis=2)
            w_vdw = jnp.where(any_match, jnp.sum(jnp.where(matches, ev_i[:, None, :], zero), axis=2), one)
            w_elec = jnp.where(any_match, jnp.sum(jnp.where(matches, ee_i[:, None, :], zero), axis=2), one)
        else:
            w_vdw = jnp.ones((chunk_size, N), dtype=dtype)
            w_elec = jnp.ones((chunk_size, N), dtype=dtype)

        sig_ij = half * (sig_i[:, None] + sig_j[None, :])
        sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
        eps_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j[None, :], zero) + jnp.asarray(1e-30, dtype=dtype)) * w_vdw

        u = dist / sig_safe
        u6 = u ** 6
        x_inv6 = one / jnp.maximum(soft_alpha_lj + u6, jnp.asarray(1e-6, dtype=dtype))
        e_lj = jnp.asarray(4.0, dtype=dtype) * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)

        dist_coul = jnp.sqrt(dist_sq + soft_alpha_coul)
        qq = q_i[:, None] * q_j[None, :] * coul_const * w_elec
        if pme_alpha > 0.0:
            e_coul = qq * erfc(alpha_val * dist_coul) / dist_coul
        else:
            e_coul = qq / dist_coul

        e_pair = jnp.where(active, e_lj + e_coul, zero)
        return acc_e + half * jnp.sum(e_pair), None

    total_e, _ = jax.lax.scan(_chunk_fn, zero, jnp.arange(n_chunks, dtype=jnp.int32))
    return total_e


@functools.partial(custom_vjp, nondiff_argnums=(10, 11, 12, 13, 14))
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
    lam_arr: jax.Array,
    has_box: bool,
    pme_alpha: float,
    softcore_alpha: float,
    softcore_exponent: int,
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
        lam_arr,
        has_box,
        pme_alpha,
        softcore_alpha,
        softcore_exponent,
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
    lam_arr,
    has_box,
    pme_alpha,
    softcore_alpha,
    softcore_exponent,
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
        lam_arr,
        has_box,
        pme_alpha,
        softcore_alpha,
        softcore_exponent,
        interpret,
    )
    return (
        energy,
        (
            forces,
            positions,
            charges,
            sigmas,
            epsilons,
            atom_mask,
            excl_indices,
            excl_scales_vdw,
            excl_scales_elec,
            box,
            lam_arr,
        ),
    )


def _pallas_flash_explicit_bwd(
    has_box,
    pme_alpha,
    softcore_alpha,
    softcore_exponent,
    interpret,
    res,
    g,
):
    del interpret
    (
        forces,
        positions,
        charges,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        lam_arr,
    ) = res
    grad_pos = -g[..., None, None] * forces if g.ndim > 0 else -g * forces

    if positions.ndim == 2:
        def _param_fn(q, sig, eps, ev, ee, lam):
            return _export_flash_one(
                positions,
                q,
                sig,
                eps,
                atom_mask,
                excl_indices,
                ev,
                ee,
                box,
                lam,
                has_box,
                pme_alpha,
                softcore_alpha,
                softcore_exponent,
            )

        _, vjp_fn = jax.vjp(_param_fn, charges, sigmas, epsilons, excl_scales_vdw, excl_scales_elec, lam_arr)
        grad_q, grad_sig, grad_eps, grad_ev, grad_ee, grad_lam = vjp_fn(g)
    else:
        B = positions.shape[0]
        b_q = charges.ndim > 1
        b_sig = sigmas.ndim > 1
        b_eps = epsilons.ndim > 1
        b_msk = atom_mask.ndim > 1
        b_ei = excl_indices.ndim > 2
        b_ev = excl_scales_vdw.ndim > 2
        b_ee = excl_scales_elec.ndim > 2
        b_box = box.ndim > 1
        b_lam = lam_arr.ndim > 1

        def _step(b):
            pos_b = positions[b]
            q_b = charges[b] if b_q else charges
            sig_b = sigmas[b] if b_sig else sigmas
            eps_b = epsilons[b] if b_eps else epsilons
            msk_b = atom_mask[b] if b_msk else atom_mask
            ei_b = excl_indices[b] if b_ei else excl_indices
            ev_b = excl_scales_vdw[b] if b_ev else excl_scales_vdw
            ee_b = excl_scales_elec[b] if b_ee else excl_scales_elec
            bx_b = box[b] if b_box else box
            lam_b = lam_arr[b] if b_lam else lam_arr
            g_b = g[b] if g.ndim > 0 else g

            def _param_fn_b(q, sig, eps, ev, ee, lam):
                return _export_flash_one(
                    pos_b,
                    q,
                    sig,
                    eps,
                    msk_b,
                    ei_b,
                    ev,
                    ee,
                    bx_b,
                    lam,
                    has_box,
                    pme_alpha,
                    softcore_alpha,
                    softcore_exponent,
                )

            _, vjp_b = jax.vjp(_param_fn_b, q_b, sig_b, eps_b, ev_b, ee_b, lam_b)
            return vjp_b(g_b)

        gq_all, gs_all, ge_all, gev_all, gee_all, glam_all = jax.lax.map(_step, jnp.arange(B))
        grad_q = gq_all if b_q else jnp.sum(gq_all, axis=0)
        grad_sig = gs_all if b_sig else jnp.sum(gs_all, axis=0)
        grad_eps = ge_all if b_eps else jnp.sum(ge_all, axis=0)
        grad_ev = gev_all if b_ev else jnp.sum(gev_all, axis=0)
        grad_ee = gee_all if b_ee else jnp.sum(gee_all, axis=0)
        grad_lam = glam_all if b_lam else jnp.sum(glam_all, axis=0)

    return (
        grad_pos,
        grad_q.astype(charges.dtype),
        grad_sig.astype(sigmas.dtype),
        grad_eps.astype(epsilons.dtype),
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(excl_indices),
        grad_ev.astype(excl_scales_vdw.dtype),
        grad_ee.astype(excl_scales_elec.dtype),
        jnp.zeros_like(box),
        grad_lam.astype(lam_arr.dtype),
    )


_pallas_flash_explicit_diff.defvjp(_pallas_flash_explicit_fwd, _pallas_flash_explicit_bwd)


def pallas_flash_explicit_nonbonded(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array | None,
    excl_indices: jax.Array | None,
    excl_scales_vdw: jax.Array | None,
    excl_scales_elec: jax.Array | None,
    box: jax.Array | None = None,
    pme_alpha: float = 0.0,
    soft_core_lambda: float | jax.Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    interpret: bool | None = None,
) -> jax.Array:
    """2D-tiled (32, 32) FlashMD explicit nonbonded Pallas kernel with forward-force custom_vjp."""
    dtype = positions.dtype
    N = positions.shape[-2]
    msk_arr = (
        jnp.ones(positions.shape[:-1], dtype=jnp.bool_)
        if atom_mask is None
        else jnp.asarray(atom_mask)
    )
    if excl_indices is None:
        ei_arr = jnp.zeros((N, 0), dtype=jnp.int32)
        ev_arr = jnp.zeros((N, 0), dtype=dtype)
        ee_arr = jnp.zeros((N, 0), dtype=dtype)
    else:
        ei_arr = jnp.asarray(excl_indices, dtype=jnp.int32)
        ev_arr = (
            jnp.zeros_like(ei_arr, dtype=dtype)
            if excl_scales_vdw is None
            else jnp.asarray(excl_scales_vdw, dtype=dtype)
        )
        ee_arr = (
            jnp.zeros_like(ei_arr, dtype=dtype)
            if excl_scales_elec is None
            else jnp.asarray(excl_scales_elec, dtype=dtype)
        )

    box_arr, has_box = _normalize_box(box, positions)
    lam_raw = jnp.asarray(soft_core_lambda, dtype=dtype)
    lam_arr = lam_raw.reshape((1,)) if lam_raw.ndim == 0 else lam_raw.reshape((-1, 1))

    if in_export_context():
        if positions.ndim == 3:
            return jax.vmap(
                lambda b: _export_flash_one(
                    positions[b],
                    charges[b] if charges.ndim == 2 else charges,
                    sigmas[b] if sigmas.ndim == 2 else sigmas,
                    epsilons[b] if epsilons.ndim == 2 else epsilons,
                    msk_arr[b] if msk_arr.ndim == 2 else msk_arr,
                    ei_arr[b] if ei_arr.ndim == 3 else ei_arr,
                    ev_arr[b] if ev_arr.ndim == 3 else ev_arr,
                    ee_arr[b] if ee_arr.ndim == 3 else ee_arr,
                    box_arr[b],
                    lam_arr[b] if lam_arr.ndim == 2 else lam_arr,
                    bool(has_box),
                    float(pme_alpha),
                    float(softcore_alpha),
                    int(softcore_exponent),
                )
            )(jnp.arange(positions.shape[0]))
        return _export_flash_one(
            positions,
            charges,
            sigmas,
            epsilons,
            msk_arr,
            ei_arr,
            ev_arr,
            ee_arr,
            box_arr,
            lam_arr,
            bool(has_box),
            float(pme_alpha),
            float(softcore_alpha),
            int(softcore_exponent),
        )

    if interpret is None:
        interpret = not _triton_gpu_available()

    return _pallas_flash_explicit_diff(
        positions,
        jnp.asarray(charges, dtype=dtype),
        jnp.asarray(sigmas, dtype=dtype),
        jnp.asarray(epsilons, dtype=dtype),
        msk_arr,
        ei_arr,
        ev_arr,
        ee_arr,
        box_arr,
        lam_arr,
        bool(has_box),
        float(pme_alpha),
        float(softcore_alpha),
        int(softcore_exponent),
        bool(interpret),
    )


__all__ = [
    "PallasKernelConfig",
    "gb_coulomb_energy_dense",
    "gb_energy_split_cutoff",
    "pallas_angles_harmonic",
    "pallas_bonds_2body",
    "pallas_born_radii",
    "pallas_cmap_torsions",
    "pallas_exceptions_14",
    "pallas_flash_explicit_nonbonded",
    "pallas_gb_ace_nl",
    "pallas_nonbonded_nl",
    "pallas_pme_exclusion_pairs",
    "pallas_sasa_nl",
    "pallas_torsions_4body",
    "resolve_pallas_mode",
]
