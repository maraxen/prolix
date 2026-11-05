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
import xtrax
from jax import custom_vjp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.scipy.special import erf, erfc
from proxide.physics import constants
from xtrax.intent import Context
from xtrax.kernel_seam import VjpKernel

xtrax_op = xtrax.op

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.generalized_born import (
    ALPHA_OBC,
    BETA_OBC,
    GAMMA_OBC,
    compute_born_radii_neighbor_list,
    compute_pair_integral,
)
from prolix.physics.neighbor_list import (
    _NB_EXCL_14_BIT,
    _NB_EXCL_FULL_BIT,
    _NB_INDEX_MASK,
    compute_morton_codes_3d,
    encode_nl_exclusions,
    get_neighbor_exclusion_scales,
    pallas_prune_and_encode_cell_list,
    permute_and_sort_exclusions,
    permute_neighbor_list_morton,
)
from prolix.physics.pallas_bonded import (
    _BLOCK_B,
    _add3,
    _bonded_all_fused_param_grads,
    _cross3,
    _dot3,
    _eval_cmap_horner_cell,
    _flatten_multi_term_torsion,
    _make_shapes_meta,
    _min_image3,
    _pl_atomic_add_1d,
    _scale3,
    _sub3,
    _torsion_angle_and_atom_grads_vec3,
    angles_harmonic_reference,
    bonded_all_fused_reference,
    bonds_2body_reference,
    cmap_torsions_reference,
    exceptions_14_reference,
    make_angles_harmonic_op,
    make_bonded_all_fused_op,
    make_bonds_2body_op,
    make_cmap_torsions_op,
    make_exceptions_14_op,
    make_pme_exclusion_pairs_op,
    make_torsions_4body_op,
    pallas_angles_harmonic,
    pallas_bonded_all_fused,
    pallas_bonds_2body,
    pallas_cmap_torsions,
    pallas_exceptions_14,
    pallas_pme_exclusion_pairs,
    pallas_torsions_4body,
    pme_exclusion_pairs_reference,
    precompute_cmap_coefficients,
    torsions_4body_reference,
)
from prolix.physics.pallas_implicit import (
    gb_ace_nl_reference,
    make_gb_ace_nl_op,
    make_sasa_nl_op,
    pallas_gb_ace_nl,
    pallas_sasa_nl,
    sasa_nl_reference,
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


def born_radii_reference(
    positions: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    neighbor_idx: jax.Array,
    *,
    dielectric_offset: float = 0.09,
) -> jax.Array:
    """Plain-JAX reference for OBC-II Born radii (2D or 3D batched)."""
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


def make_born_radii_op(
    *,
    dielectric_offset: float = 0.09,
    context: Context | None = None,
):
    """Build an ``xtrax.op`` for OBC-II Born radii."""
    d_off = float(dielectric_offset)
    ref = functools.partial(born_radii_reference, dielectric_offset=d_off)

    def _make_vjp_kernel(*, interpret: bool) -> VjpKernel:
        def _fwd(pos, rad, sc_rad, nbr):
            if pos.ndim == 3:
                B = pos.shape[0]
                b_rad = rad.ndim > 1
                b_sc = sc_rad.ndim > 1
                b_nbr = nbr.ndim > 2
                out = jax.vmap(
                    lambda b: _pallas_born_radii_vmappable(
                        pos[b],
                        rad[b] if b_rad else rad,
                        sc_rad[b] if b_sc else sc_rad,
                        nbr[b] if b_nbr else nbr,
                        d_off,
                        interpret,
                    )
                )(jnp.arange(B))
                return out, (pos, rad, sc_rad, nbr)
            return _pallas_born_radii_fwd(pos, rad, sc_rad, d_off, nbr, interpret)

        def _bwd(res, g):
            return _pallas_born_radii_bwd(d_off, interpret, res, g)

        return VjpKernel(fwd=_fwd, bwd=_bwd)

    return xtrax_op(
        ref,
        vjp_kernels={
            "gpu": _make_vjp_kernel(interpret=False),
            "cpu": _make_vjp_kernel(interpret=True),
        },
        name="prolix.born_radii",
        context=context,
    )


def pallas_born_radii(
    positions: jax.Array,
    radii: jax.Array,
    neighbor_idx: jax.Array,
    dielectric_offset: float = 0.09,
    scaled_radii: jax.Array | None = None,
    interpret: bool | None = None,
) -> jax.Array:
    """Computes OBC-II Born radii using a vectorized (32, 32) Pallas kernel.

    Supports ``jax.grad`` (via ``xtrax.op`` ``VjpKernel``), ``jax.vmap`` (via ``custom_vmap``),
    and automatic ``interpret=True`` fallback on CPU.
    """
    if scaled_radii is None:
        scaled_radii = radii
    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
    else:
        if interpret is None:
            interpret = not _triton_gpu_available()
        ctx = Context(kernels=True, backend="cpu" if interpret else "gpu")
    op = make_born_radii_op(dielectric_offset=float(dielectric_offset), context=ctx)
    return op(positions, radii, scaled_radii, neighbor_idx)


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


def _pallas_nonbonded_nl_batch(
    axis_size: int,
    in_batched: tuple[bool, ...],
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
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> tuple[jax.Array, jax.Array]:
    """Batched 2D-grid (B, n_blocks) Pallas O(N*K) nonbonded energy & forces."""
    B = int(axis_size)
    pos_b = positions if in_batched[0] else positions[None, ...]
    q_b = charges if in_batched[1] else charges[None, ...]
    sig_b = sigmas if in_batched[2] else sigmas[None, ...]
    eps_b = epsilons if in_batched[3] else epsilons[None, ...]
    mask_b = atom_mask if in_batched[4] else atom_mask[None, ...]
    nbr_b = neighbor_idx if in_batched[5] else neighbor_idx[None, ...]
    ei_b = excl_indices if in_batched[6] else excl_indices[None, ...]
    ev_b = excl_scales_vdw if in_batched[7] else excl_scales_vdw[None, ...]
    ee_b = excl_scales_elec if in_batched[8] else excl_scales_elec[None, ...]
    bx_b = box if in_batched[9] else box[None, ...]
    lam_b = lam_arr.reshape((B if in_batched[10] else 1, -1))[:, 0]

    B_pos = pos_b.shape[0]
    B_q = q_b.shape[0]
    B_sig = sig_b.shape[0]
    B_eps = eps_b.shape[0]
    B_mask = mask_b.shape[0]
    B_nbr = nbr_b.shape[0]
    B_ei = ei_b.shape[0]
    B_ev = ev_b.shape[0]
    B_ee = ee_b.shape[0]
    B_box = bx_b.shape[0]
    B_lam = lam_b.shape[0]

    N, K = int(nbr_b.shape[1]), int(nbr_b.shape[2])
    M = int(ei_b.shape[2])
    if M > _NB_MAX_EXCL:
        raise ValueError(f"excl_indices width {M} exceeds _NB_MAX_EXCL={_NB_MAX_EXCL}")
    orig_dtype = pos_b.dtype
    if N == 0 or K == 0:
        return (
            jnp.zeros((B,), dtype=orig_dtype),
            jnp.zeros((B, N, 3), dtype=orig_dtype),
        )
    dtype = jnp.float32
    pos_b = pos_b.astype(dtype)
    q_b = q_b.astype(dtype)
    sig_b = sig_b.astype(dtype)
    eps_b = eps_b.astype(dtype)
    use_pre_encoded = excl_mode == "pre_encoded"

    n_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    N_pad = n_blocks * _NB_BLOCK_N
    pad_n = N_pad - N
    k_blocks = max(1, (K + _NB_BLOCK_K - 1) // _NB_BLOCK_K)

    box_vec_b = jnp.asarray(bx_b.astype(dtype))
    if has_box:
        L_xyz = box_vec_b[:, :3]
        tilt_xyz = box_vec_b[:, 3:6]
        is_ortho_box = jnp.all(tilt_xyz == jnp.asarray(0.0, dtype=dtype), axis=-1, keepdims=True)
        safe_L = jnp.where(L_xyz > jnp.asarray(0.0, dtype=dtype), L_xyz, jnp.ones_like(L_xyz))
        pos_wrapped = jnp.where(
            (L_xyz[:, None, :] > jnp.asarray(0.0, dtype=dtype)) & is_ortho_box[:, None, :],
            pos_b - L_xyz[:, None, :] * jnp.floor(pos_b / safe_L[:, None, :]),
            pos_b,
        )
    else:
        pos_wrapped = pos_b

    B_pos = pos_wrapped.shape[0]
    if pad_n > 0:
        pos_pad = jnp.pad(pos_wrapped, ((0, 0), (0, pad_n), (0, 0)))
        q_pad = jnp.pad(q_b, ((0, 0), (0, pad_n)))
        sig_pad = jnp.pad(sig_b, ((0, 0), (0, pad_n)), constant_values=1.0)
        eps_pad = jnp.pad(eps_b, ((0, 0), (0, pad_n)))
        mask_pad = jnp.pad(mask_b.astype(jnp.int32), ((0, 0), (0, pad_n)), constant_values=0)
    else:
        pos_pad = pos_wrapped
        q_pad = q_b
        sig_pad = sig_b
        eps_pad = eps_b
        mask_pad = mask_b.astype(jnp.int32)

    if use_pre_encoded:
        excl_min_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_max_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_idx_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_vdw_flat = jnp.zeros((1,), dtype=dtype)
        excl_elec_flat = jnp.zeros((1,), dtype=dtype)
    else:
        pad_m = max(0, _NB_MAX_EXCL - M)
        pad_n_excl = N_pad - ei_b.shape[1]
        excl_idx_pad = ei_b[:, :, :_NB_MAX_EXCL].astype(jnp.int32)
        excl_vdw_pad = ev_b[:, :, :_NB_MAX_EXCL].astype(dtype)
        excl_elec_pad = ee_b[:, :, :_NB_MAX_EXCL].astype(dtype)
        if pad_n_excl > 0 or pad_m > 0:
            excl_idx_pad = jnp.pad(excl_idx_pad, ((0, 0), (0, pad_n_excl), (0, pad_m)), constant_values=-1)
            excl_vdw_pad = jnp.pad(excl_vdw_pad, ((0, 0), (0, pad_n_excl), (0, pad_m)), constant_values=1.0)
            excl_elec_pad = jnp.pad(excl_elec_pad, ((0, 0), (0, pad_n_excl), (0, pad_m)), constant_values=1.0)
        int_max_i32 = jnp.int32(0x7FFFFFFF)
        valid_e_pad = (excl_idx_pad >= 0) & (excl_idx_pad < N)
        excl_min_pad = jnp.min(jnp.where(valid_e_pad, excl_idx_pad, int_max_i32), axis=-1)
        excl_max_pad = jnp.max(jnp.where(valid_e_pad, excl_idx_pad, jnp.int32(-1)), axis=-1)
        excl_min_flat = jnp.asarray(excl_min_pad.reshape(B_ei * N_pad))
        excl_max_flat = jnp.asarray(excl_max_pad.reshape(B_ei * N_pad))
        excl_idx_flat = jnp.asarray(excl_idx_pad.reshape(B_ei * N_pad * _NB_MAX_EXCL))
        excl_vdw_flat = jnp.asarray(excl_vdw_pad.reshape(B_ev * N_pad * _NB_MAX_EXCL))
        excl_elec_flat = jnp.asarray(excl_elec_pad.reshape(B_ee * N_pad * _NB_MAX_EXCL))

    pos_x = jnp.asarray(pos_pad[:, :, 0].reshape(B_pos * N_pad))
    pos_y = jnp.asarray(pos_pad[:, :, 1].reshape(B_pos * N_pad))
    pos_z = jnp.asarray(pos_pad[:, :, 2].reshape(B_pos * N_pad))
    q_flat = jnp.asarray(q_pad.reshape(B_q * N_pad))
    sig_flat = jnp.asarray(sig_pad.reshape(B_sig * N_pad))
    eps_flat = jnp.asarray(eps_pad.reshape(B_eps * N_pad))
    mask_flat = jnp.asarray(mask_pad.reshape(B_mask * N_pad))
    nbr_flat = jnp.asarray(nbr_b.astype(jnp.int32).reshape(B_nbr * N * K))
    box_flat = jnp.asarray(box_vec_b.reshape(B_box * 6))
    lam_flat = jnp.asarray(lam_b.astype(dtype).reshape(B_lam))
    r_margin = float(max(cutoff, 6.0))

    def _nb_nl_kernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        sig_ref,
        eps_ref,
        mask_ref,
        nbr_flat_ref,
        excl_min_ref,
        excl_max_ref,
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
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        offs_i = pid * _NB_BLOCK_N + jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
        in_bounds_i = offs_i < N
        safe_offs_i = jnp.minimum(offs_i, max(N - 1, 0))

        base_pos = (b_id * N_pad) if B_pos > 1 else jnp.int32(0)
        base_q = (b_id * N_pad) if B_q > 1 else jnp.int32(0)
        base_sig = (b_id * N_pad) if B_sig > 1 else jnp.int32(0)
        base_eps = (b_id * N_pad) if B_eps > 1 else jnp.int32(0)
        base_mask = (b_id * N_pad) if B_mask > 1 else jnp.int32(0)
        base_nbr = (b_id * (N * K)) if B_nbr > 1 else jnp.int32(0)
        base_box = (b_id * 6) if B_box > 1 else jnp.int32(0)
        base_lam = b_id if B_lam > 1 else jnp.int32(0)
        base_out = b_id * N_pad

        mask_i = (_pl_load(mask_ref, (base_mask + offs_i,)) > 0) & in_bounds_i
        px_i = _pl_load(px_ref, (base_pos + offs_i,))
        py_i = _pl_load(py_ref, (base_pos + offs_i,))
        pz_i = _pl_load(pz_ref, (base_pos + offs_i,))
        q_i = _pl_load(q_ref, (base_q + offs_i,))
        sig_i = _pl_load(sig_ref, (base_sig + offs_i,))
        eps_i = _pl_load(eps_ref, (base_eps + offs_i,))

        if not use_pre_encoded:
            base_ei = (b_id * N_pad) if B_ei > 1 else jnp.int32(0)
            base_ei_flat = (b_id * (N_pad * _NB_MAX_EXCL)) if B_ei > 1 else jnp.int32(0)
            base_ev_flat = (b_id * (N_pad * _NB_MAX_EXCL)) if B_ev > 1 else jnp.int32(0)
            base_ee_flat = (b_id * (N_pad * _NB_MAX_EXCL)) if B_ee > 1 else jnp.int32(0)
            excl_min_i = _pl_load(excl_min_ref, (base_ei + offs_i,))
            excl_max_i = _pl_load(excl_max_ref, (base_ei + offs_i,))
            offs_m = jnp.arange(_NB_MAX_EXCL, dtype=jnp.int32)
            flat_excl_offs = (offs_i[:, None] * _NB_MAX_EXCL + offs_m[None, :]).reshape(
                _NB_BLOCK_N * _NB_MAX_EXCL
            )

        ax = _pl_load(box_ref, (base_box + jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (base_box + jnp.ones((1,), dtype=jnp.int32),))[0]
        cz = _pl_load(box_ref, (base_box + jnp.full((1,), 2, dtype=jnp.int32),))[0]
        bx_t = _pl_load(box_ref, (base_box + jnp.full((1,), 3, dtype=jnp.int32),))[0]
        cx_t = _pl_load(box_ref, (base_box + jnp.full((1,), 4, dtype=jnp.int32),))[0]
        cy_t = _pl_load(box_ref, (base_box + jnp.full((1,), 5, dtype=jnp.int32),))[0]

        zero_sc = jnp.asarray(0.0, dtype=dtype)
        one_sc = jnp.asarray(1.0, dtype=dtype)
        s14_vdw = jnp.asarray(scale_14_vdw, dtype=dtype)
        s14_elec = jnp.asarray(scale_14_elec, dtype=dtype)

        if has_box:
            pos_big = jnp.asarray(1e30, dtype=dtype)
            neg_big = jnp.asarray(-1e30, dtype=dtype)
            r_min_x = jnp.min(jnp.where(mask_i, px_i, pos_big))
            r_max_x = jnp.max(jnp.where(mask_i, px_i, neg_big))
            r_min_y = jnp.min(jnp.where(mask_i, py_i, pos_big))
            r_max_y = jnp.max(jnp.where(mask_i, py_i, neg_big))
            r_min_z = jnp.min(jnp.where(mask_i, pz_i, pos_big))
            r_max_z = jnp.max(jnp.where(mask_i, pz_i, neg_big))
            is_ortho = (bx_t == zero_sc) & (cx_t == zero_sc) & (cy_t == zero_sc)
            r_m_val = jnp.asarray(r_margin, dtype=dtype)
            interior_x = is_ortho & (r_min_x >= r_m_val) & (r_max_x <= ax - r_m_val)
            interior_y = is_ortho & (r_min_y >= r_m_val) & (r_max_y <= by - r_m_val)
            interior_z = is_ortho & (r_min_z >= r_m_val) & (r_max_z <= cz - r_m_val)
            interior_all = interior_x & interior_y & interior_z
        else:
            interior_all = jnp.bool_(True)

        lam = _pl_load(lam_ref, (base_lam + jnp.zeros((1,), dtype=jnp.int32),))[0]
        one_minus_lam = one_sc - lam
        soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (
            one_minus_lam ** int(softcore_exponent)
        )
        coul_const = jnp.asarray(coulomb_constant, dtype=dtype)
        alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
        two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

        def _k_tile_body(k_tile, carry):
            offs_k = k_tile * _NB_BLOCK_K + jnp.arange(_NB_BLOCK_K, dtype=jnp.int32)
            in_bounds_nk = in_bounds_i[:, None] & (offs_k[None, :] < K)
            safe_offs_k = jnp.minimum(offs_k, max(K - 1, 0))
            flat_nbr_offs = base_nbr + (safe_offs_i[:, None] * K + safe_offs_k[None, :]).reshape(
                _NB_BLOCK_N * _NB_BLOCK_K
            )
            raw_j_1d = _pl_load(nbr_flat_ref, (flat_nbr_offs,))
            raw_j = jnp.where(in_bounds_nk, raw_j_1d.reshape(_NB_BLOCK_N, _NB_BLOCK_K), jnp.int32(N))
            is_full_excl = (raw_j >= 0) & ((raw_j & jnp.int32(_NB_EXCL_FULL_BIT)) != 0)
            is_14_excl = (raw_j >= 0) & ((raw_j & jnp.int32(_NB_EXCL_14_BIT)) != 0)
            j_idx = jnp.where(raw_j >= 0, raw_j & jnp.int32(_NB_INDEX_MASK), raw_j)
            cand_pair = (
                mask_i[:, None]
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
                & (True if (not use_pre_encoded or include_pme_excl_erf) else (~is_full_excl))
            )
            has_any_valid = jnp.max(cand_pair.astype(jnp.int32)) > 0

            def _eval_nonempty_tile(c_in):
                acc_e, acc_fx, acc_fy, acc_fz = c_in
                safe_j_1d = jnp.clip(j_idx.reshape(_NB_BLOCK_N * _NB_BLOCK_K), 0, N_pad - 1)

                px_j = _pl_load(px_ref, (base_pos + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                py_j = _pl_load(py_ref, (base_pos + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                pz_j = _pl_load(pz_ref, (base_pos + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                q_j = _pl_load(q_ref, (base_q + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                sig_j = _pl_load(sig_ref, (base_sig + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                eps_j = _pl_load(eps_ref, (base_eps + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                mask_j = _pl_load(mask_ref, (base_mask + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K) > 0

                valid_pair = cand_pair & mask_j

                raw_dx = px_i[:, None] - px_j
                raw_dy = py_i[:, None] - py_j
                raw_dz = pz_i[:, None] - pz_j
                if has_box:
                    dx, dy, dz = jax.lax.cond(
                        interior_all,
                        lambda d: d,
                        lambda d: _min_image_triclinic_xyz(
                            d[0], d[1], d[2], ax, by, cz, bx_t, cx_t, cy_t, True
                        ),
                        (raw_dx, raw_dy, raw_dz),
                    )
                else:
                    dx, dy, dz = raw_dx, raw_dy, raw_dz

                r_sq = dx * dx + dy * dy + dz * dz + jnp.asarray(1e-12, dtype=dtype)
                dist = jnp.sqrt(r_sq)
                in_cutoff = dist < jnp.asarray(cutoff, dtype=dtype)
                active = valid_pair & in_cutoff

                zero = jnp.asarray(0.0, dtype=dtype)
                if use_pre_encoded:
                    w_vdw = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, one_sc))
                    w_elec = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_elec, one_sc))
                else:
                    has_tag = is_full_excl | is_14_excl
                    in_excl_range = (
                        active
                        & (~has_tag)
                        & (j_idx >= excl_min_i[:, None])
                        & (j_idx <= excl_max_i[:, None])
                    )

                    def _lookup_excl_tile(_):
                        excl_idx_i = _pl_load(excl_idx_flat_ref, (base_ei_flat + flat_excl_offs,)).reshape(
                            _NB_BLOCK_N, _NB_MAX_EXCL
                        )
                        excl_vdw_i = _pl_load(excl_vdw_flat_ref, (base_ev_flat + flat_excl_offs,)).reshape(
                            _NB_BLOCK_N, _NB_MAX_EXCL
                        )
                        excl_elec_i = _pl_load(excl_elec_flat_ref, (base_ee_flat + flat_excl_offs,)).reshape(
                            _NB_BLOCK_N, _NB_MAX_EXCL
                        )
                        valid_excl_i = (excl_idx_i >= 0) & (excl_idx_i < N)
                        matches = (j_idx[:, :, None] == excl_idx_i[:, None, :]) & valid_excl_i[:, None, :]
                        any_match = jnp.sum(matches.astype(jnp.int32), axis=2) > 0
                        wv = jnp.where(
                            any_match,
                            jnp.sum(jnp.where(matches, excl_vdw_i[:, None, :], zero), axis=2),
                            one_sc,
                        )
                        we = jnp.where(
                            any_match,
                            jnp.sum(jnp.where(matches, excl_elec_i[:, None, :], zero), axis=2),
                            one_sc,
                        )
                        return wv, we

                    def _no_excl_tile(_):
                        ones_t = jnp.ones((_NB_BLOCK_N, _NB_BLOCK_K), dtype=dtype)
                        return ones_t, ones_t

                    w_vdw_tbl, w_elec_tbl = jax.lax.cond(
                        jnp.sum(in_excl_range.astype(jnp.int32)) > 0,
                        _lookup_excl_tile,
                        _no_excl_tile,
                        None,
                    )
                    w_vdw = jnp.where(
                        is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, w_vdw_tbl)
                    )
                    w_elec = jnp.where(
                        is_full_excl, zero, jnp.where(is_14_excl, s14_elec, w_elec_tbl)
                    )

                sig_ij = jnp.asarray(0.5, dtype=dtype) * (sig_i[:, None] + sig_j)
                sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), jnp.asarray(1.0, dtype=dtype))
                eps_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j, jnp.asarray(0.0, dtype=dtype)) + jnp.asarray(1e-30, dtype=dtype))

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
                f_scalar = jnp.where(active, (zero - (de_lj_dr + de_coul_dr)) / dist, zero)

                acc_e = acc_e + jnp.asarray(0.5, dtype=dtype) * jnp.sum(e_pair, axis=1)
                acc_fx = acc_fx + jnp.sum(f_scalar * dx, axis=1)
                acc_fy = acc_fy + jnp.sum(f_scalar * dy, axis=1)
                acc_fz = acc_fz + jnp.sum(f_scalar * dz, axis=1)
                return (acc_e, acc_fx, acc_fy, acc_fz)

            return jax.lax.cond(has_any_valid, _eval_nonempty_tile, lambda c_in: c_in, carry)

        def _run_k_tiles(c0):
            return jax.lax.fori_loop(0, k_blocks, _k_tile_body, c0)

        init_carry = (
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
        )
        e_blk, fx_blk, fy_blk, fz_blk = jax.lax.cond(
            jnp.max(mask_i.astype(jnp.int32)) > 0,
            _run_k_tiles,
            lambda c0: c0,
            init_carry,
        )
        _pl_store(e_out_ref, (base_out + offs_i,), e_blk)
        _pl_store(fx_out_ref, (base_out + offs_i,), fx_blk)
        _pl_store(fy_out_ref, (base_out + offs_i,), fy_blk)
        _pl_store(fz_out_ref, (base_out + offs_i,), fz_blk)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(15)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [jax.ShapeDtypeStruct((B * N_pad,), dtype) for _ in range(4)]

    e_pad, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _nb_nl_kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(B, n_blocks),
        interpret=interpret,
        compiler_params=(
            resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=B * n_blocks)
            if not interpret
            else None
        ),
        name="pallas_nonbonded_nl",
    )(
        pos_x,
        pos_y,
        pos_z,
        q_flat,
        sig_flat,
        eps_flat,
        mask_flat,
        nbr_flat,
        excl_min_flat,
        excl_max_flat,
        excl_idx_flat,
        excl_vdw_flat,
        excl_elec_flat,
        box_flat,
        lam_flat,
    )
    e_b = jnp.sum(e_pad.reshape(B, N_pad)[:, :N].astype(orig_dtype), axis=-1)
    forces_b = jnp.stack(
        [
            fx_pad.reshape(B, N_pad)[:, :N],
            fy_pad.reshape(B, N_pad)[:, :N],
            fz_pad.reshape(B, N_pad)[:, :N],
        ],
        axis=-1,
    ).astype(orig_dtype)
    return e_b, forces_b


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
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> tuple[jax.Array, jax.Array]:
    """Single-system vectorized (32, 32) Pallas O(N*K) nonbonded energy & forces (v2 O1 & O4)."""
    e_b, f_b = _pallas_nonbonded_nl_batch(
        1,
        (False,) * 11,
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
        excl_mode,
        scale_14_vdw,
        scale_14_elec,
    )
    return e_b[0], f_b[0]


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
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
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
    em = str(excl_mode)
    s14_v = float(scale_14_vdw)
    s14_e = float(scale_14_elec)

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
            em,
            s14_v,
            s14_e,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, sig, eps, mask, nbr, ei, ev, ee, bx, lam):
        n_nbr = int(nbr.shape[-2]) * int(nbr.shape[-1])
        b_nbr = int(axis_size) if in_batched[5] else 1
        if b_nbr * n_nbr < (1 << 30):
            e_batch, f_batch = _pallas_nonbonded_nl_batch(
                int(axis_size),
                tuple(bool(b) for b in in_batched),
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
                em,
                s14_v,
                s14_e,
            )
            return (e_batch, f_batch), (True, True)

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

    if positions.ndim == 3:
        B = int(positions.shape[0])
        in_batched = (
            True,
            charges.ndim == 2,
            sigmas.ndim == 2,
            epsilons.ndim == 2,
            atom_mask.ndim == 2,
            neighbor_idx.ndim == 3,
            excl_indices.ndim == 3,
            excl_scales_vdw.ndim == 3,
            excl_scales_elec.ndim == 3,
            box.ndim == 2,
            lam_arr.ndim >= 2,
        )
        n_nbr = int(neighbor_idx.shape[-2]) * int(neighbor_idx.shape[-1])
        b_nbr = B if in_batched[5] else 1
        if b_nbr * n_nbr < (1 << 30):
            return _pallas_nonbonded_nl_batch(
                B,
                in_batched,
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
                hb,
                pa,
                co,
                cc,
                sw,
                sca,
                sce,
                inc_erf,
                interp,
                em,
                s14_v,
                s14_e,
            )
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                charges[b] if charges.ndim == 2 else charges,
                sigmas[b] if sigmas.ndim == 2 else sigmas,
                epsilons[b] if epsilons.ndim == 2 else epsilons,
                atom_mask[b] if atom_mask.ndim == 2 else atom_mask,
                neighbor_idx[b] if neighbor_idx.ndim == 3 else neighbor_idx,
                excl_indices[b] if excl_indices.ndim == 3 else excl_indices,
                excl_scales_vdw[b] if excl_scales_vdw.ndim == 3 else excl_scales_vdw,
                excl_scales_elec[b] if excl_scales_elec.ndim == 3 else excl_scales_elec,
                box[b] if box.ndim == 2 else box,
                lam_arr[b] if lam_arr.ndim == 2 else lam_arr,
            )
        )(jnp.arange(B))

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
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Chunked O(chunk_size*K) analytical parameter gradients (grad_q, grad_sig, grad_eps, grad_excl_vdw, grad_excl_elec, grad_lam)."""
    N, K = neighbor_idx.shape
    M_excl = excl_indices.shape[1]
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    s14_vdw = jnp.asarray(scale_14_vdw, dtype=dtype)
    s14_elec = jnp.asarray(scale_14_elec, dtype=dtype)

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
        raw_nbr_i = jax.lax.dynamic_slice(nbr_pad, (i_start, z_i32), (chunk_size, K))
        ei_i = jax.lax.dynamic_slice(ei_pad, (i_start, z_i32), (chunk_size, M_excl))
        ev_i = jax.lax.dynamic_slice(ev_pad, (i_start, z_i32), (chunk_size, M_excl))
        ee_i = jax.lax.dynamic_slice(ee_pad, (i_start, z_i32), (chunk_size, M_excl))

        is_full_excl = (raw_nbr_i >= 0) & ((raw_nbr_i & jnp.int32(_NB_EXCL_FULL_BIT)) != 0)
        is_14_excl = (raw_nbr_i >= 0) & ((raw_nbr_i & jnp.int32(_NB_EXCL_14_BIT)) != 0)
        nbr_i = jnp.where(raw_nbr_i >= 0, raw_nbr_i & jnp.int32(_NB_INDEX_MASK), raw_nbr_i)

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
            w_vdw_tbl = jnp.where(any_match, jnp.sum(jnp.where(matches, ev_i[:, None, :], zero), axis=2), one)
            w_elec_tbl = jnp.where(any_match, jnp.sum(jnp.where(matches, ee_i[:, None, :], zero), axis=2), one)
        else:
            matches = jnp.zeros((chunk_size, K, 0), dtype=jnp.bool_)
            w_vdw_tbl = jnp.ones((chunk_size, K), dtype=dtype)
            w_elec_tbl = jnp.ones((chunk_size, K), dtype=dtype)

        w_vdw = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, w_vdw_tbl))
        w_elec = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_elec, w_elec_tbl))

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


def _unwrap_primal(x):
    return x.value if hasattr(x, "perturbed") else x


def _is_perturbed(x, default: bool = True) -> bool:
    return bool(x.perturbed) if hasattr(x, "perturbed") else bool(default)


@functools.partial(custom_vjp, nondiff_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23))
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
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    compute_param_grads: bool | None = None,
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
        excl_mode,
        scale_14_vdw,
        scale_14_elec,
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
    excl_mode="table",
    scale_14_vdw=0.5,
    scale_14_elec=0.83333333,
    compute_param_grads=None,
):
    pos_v = _unwrap_primal(positions)
    q_v = _unwrap_primal(charges)
    sig_v = _unwrap_primal(sigmas)
    eps_v = _unwrap_primal(epsilons)
    msk_v = _unwrap_primal(atom_mask)
    nbr_v = _unwrap_primal(neighbor_idx)
    ei_v = _unwrap_primal(excl_indices)
    ev_v = _unwrap_primal(excl_scales_vdw)
    ee_v = _unwrap_primal(excl_scales_elec)
    bx_v = _unwrap_primal(box)
    lam_v = _unwrap_primal(lam_arr)

    energy, forces = _pallas_nonbonded_nl_vmappable(
        pos_v,
        q_v,
        sig_v,
        eps_v,
        msk_v,
        nbr_v,
        ei_v,
        ev_v,
        ee_v,
        bx_v,
        lam_v,
        has_box,
        pme_alpha,
        cutoff,
        coulomb_constant,
        lj_switch_width,
        softcore_alpha,
        softcore_exponent,
        include_pme_excl_erf,
        interpret,
        excl_mode,
        scale_14_vdw,
        scale_14_elec,
    )
    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    if compute_param_grads is not None:
        any_param_perturbed = bool(compute_param_grads)
    else:
        any_param_perturbed = (
            _is_perturbed(charges, default_pert)
            or _is_perturbed(sigmas, default_pert)
            or _is_perturbed(epsilons, default_pert)
            or _is_perturbed(excl_scales_vdw, default_pert)
            or _is_perturbed(excl_scales_elec, default_pert)
            or _is_perturbed(lam_arr, default_pert)
        )

    shapes_meta = _make_shapes_meta(
        pos_v, q_v, sig_v, eps_v, msk_v, nbr_v, ei_v, ev_v, ee_v, bx_v, lam_v
    )
    param_res = (
        (pos_v, q_v, sig_v, eps_v, msk_v, nbr_v, ei_v, ev_v, ee_v, bx_v, lam_v)
        if any_param_perturbed
        else None
    )
    return energy, (forces, shapes_meta, param_res)


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
    excl_mode,
    scale_14_vdw,
    scale_14_elec,
    compute_param_grads,
    res,
    g,
):
    del interpret, excl_mode, compute_param_grads
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
    ) = param_res

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
            scale_14_vdw,
            scale_14_elec,
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
                scale_14_vdw,
                scale_14_elec,
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


_pallas_nonbonded_nl_diff.defvjp(
    _pallas_nonbonded_nl_fwd, _pallas_nonbonded_nl_bwd, symbolic_zeros=True
)


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
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> jax.Array:
    """Portable XLA fallback for pallas_nonbonded_nl under export_context()."""
    N, K = neighbor_idx.shape
    dtype = positions.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    if N == 0 or K == 0:
        return zero

    raw_nbr = neighbor_idx.astype(jnp.int32)
    is_full_excl = (raw_nbr >= 0) & ((raw_nbr & jnp.int32(_NB_EXCL_FULL_BIT)) != 0)
    is_14_excl = (raw_nbr >= 0) & ((raw_nbr & jnp.int32(_NB_EXCL_14_BIT)) != 0)
    nbr_clean = jnp.where(raw_nbr >= 0, raw_nbr & jnp.int32(_NB_INDEX_MASK), raw_nbr)

    safe_j = jnp.clip(nbr_clean, 0, max(N - 1, 0))
    dr = _min_image_triclinic_vec(positions[:, None, :] - positions[safe_j], box, has_box)

    dist = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=dtype))
    m_bool = jnp.asarray(atom_mask, dtype=jnp.bool_)
    active = (
        m_bool[:, None]
        & m_bool[safe_j]
        & (nbr_clean >= 0)
        & (nbr_clean < N)
        & (nbr_clean != jnp.arange(N, dtype=jnp.int32)[:, None])
        & (dist < jnp.asarray(cutoff, dtype=dtype))
    )
    w_vdw_tbl, w_elec_tbl = get_neighbor_exclusion_scales(
        excl_indices, excl_scales_vdw, excl_scales_elec, nbr_clean
    )
    w_vdw = jnp.where(
        is_full_excl,
        zero,
        jnp.where(is_14_excl, jnp.asarray(scale_14_vdw, dtype=dtype), w_vdw_tbl.astype(dtype)),
    )
    w_elec = jnp.where(
        is_full_excl,
        zero,
        jnp.where(is_14_excl, jnp.asarray(scale_14_elec, dtype=dtype), w_elec_tbl.astype(dtype)),
    )

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

    return half * jnp.sum(jnp.where(active, e_lj + e_coul, zero)).astype(dtype)


def nonbonded_nl_reference(
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
    *,
    has_box: bool = False,
    pme_alpha: float = 0.0,
    cutoff: float = 10.0,
    coulomb_constant: float = constants.COULOMB_CONSTANT,
    lj_switch_width: float = 0.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    include_pme_excl_erf: bool = True,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> jax.Array:
    """Plain-JAX reference for O(N*K) neighbor-list nonbonded energy (2D or 3D batched)."""
    if positions.ndim == 3:
        return jax.vmap(
            lambda b: _export_nonbonded_nl_one(
                positions[b],
                charges[b] if charges.ndim == 2 else charges,
                sigmas[b] if sigmas.ndim == 2 else sigmas,
                epsilons[b] if epsilons.ndim == 2 else epsilons,
                atom_mask[b] if atom_mask.ndim == 2 else atom_mask,
                neighbor_idx[b] if neighbor_idx.ndim == 3 else neighbor_idx,
                excl_indices[b] if excl_indices.ndim == 3 else excl_indices,
                excl_scales_vdw[b] if excl_scales_vdw.ndim == 3 else excl_scales_vdw,
                excl_scales_elec[b] if excl_scales_elec.ndim == 3 else excl_scales_elec,
                box[b] if box.ndim == 2 else box,
                lam_arr[b] if lam_arr.ndim == 2 else lam_arr,
                bool(has_box),
                float(pme_alpha),
                float(cutoff),
                float(coulomb_constant),
                float(lj_switch_width),
                float(softcore_alpha),
                int(softcore_exponent),
                bool(include_pme_excl_erf),
                float(scale_14_vdw),
                float(scale_14_elec),
            )
        )(jnp.arange(positions.shape[0]))
    return _export_nonbonded_nl_one(
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
        bool(has_box),
        float(pme_alpha),
        float(cutoff),
        float(coulomb_constant),
        float(lj_switch_width),
        float(softcore_alpha),
        int(softcore_exponent),
        bool(include_pme_excl_erf),
        float(scale_14_vdw),
        float(scale_14_elec),
    )


def make_nonbonded_nl_op(
    *,
    has_box: bool = False,
    pme_alpha: float = 0.0,
    cutoff: float = 10.0,
    coulomb_constant: float = constants.COULOMB_CONSTANT,
    lj_switch_width: float = 0.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    include_pme_excl_erf: bool = True,
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    compute_param_grads: bool | None = None,
    context: Context | None = None,
):
    """Build an ``xtrax.op`` for O(N*K) neighbor-list nonbonded energy."""
    hb = bool(has_box)
    pa = float(pme_alpha)
    co = float(cutoff)
    cc = float(coulomb_constant)
    sw = float(lj_switch_width)
    sca = float(softcore_alpha)
    sce = int(softcore_exponent)
    inc_erf = bool(include_pme_excl_erf)
    em = str(excl_mode)
    s14_v = float(scale_14_vdw)
    s14_e = float(scale_14_elec)
    cpg = None if compute_param_grads is None else bool(compute_param_grads)
    ref = functools.partial(
        nonbonded_nl_reference,
        has_box=hb,
        pme_alpha=pa,
        cutoff=co,
        coulomb_constant=cc,
        lj_switch_width=sw,
        softcore_alpha=sca,
        softcore_exponent=sce,
        include_pme_excl_erf=inc_erf,
        scale_14_vdw=s14_v,
        scale_14_elec=s14_e,
    )

    def _make_vjp_kernel(*, interpret: bool) -> VjpKernel:
        def _fwd(pos, q, sig, eps, msk, nbr, ei, ev, ee, bx, lam):
            if pos.ndim == 3:
                B = pos.shape[0]
                e, f = jax.vmap(
                    lambda b: _pallas_nonbonded_nl_vmappable(
                        pos[b],
                        q[b] if q.ndim == 2 else q,
                        sig[b] if sig.ndim == 2 else sig,
                        eps[b] if eps.ndim == 2 else eps,
                        msk[b] if msk.ndim == 2 else msk,
                        nbr[b] if nbr.ndim == 3 else nbr,
                        ei[b] if ei.ndim == 3 else ei,
                        ev[b] if ev.ndim == 3 else ev,
                        ee[b] if ee.ndim == 3 else ee,
                        bx[b] if bx.ndim == 2 else bx,
                        lam[b] if lam.ndim == 2 else lam,
                        hb,
                        pa,
                        co,
                        cc,
                        sw,
                        sca,
                        sce,
                        inc_erf,
                        interpret,
                        em,
                        s14_v,
                        s14_e,
                    )
                )(jnp.arange(B))
                shapes_meta = _make_shapes_meta(
                    pos, q, sig, eps, msk, nbr, ei, ev, ee, bx, lam
                )
                param_res = (
                    (pos, q, sig, eps, msk, nbr, ei, ev, ee, bx, lam)
                    if (cpg is None or cpg)
                    else None
                )
                return e, (f, shapes_meta, param_res)
            return _pallas_nonbonded_nl_fwd(
                pos, q, sig, eps, msk, nbr, ei, ev, ee, bx, lam,
                hb, pa, co, cc, sw, sca, sce, inc_erf, interpret,
                em, s14_v, s14_e, cpg,
            )

        def _bwd(res, g):
            return _pallas_nonbonded_nl_bwd(
                hb, pa, co, cc, sw, sca, sce, inc_erf, interpret,
                em, s14_v, s14_e, cpg, res, g
            )

        return VjpKernel(fwd=_fwd, bwd=_bwd)

    return xtrax_op(
        ref,
        vjp_kernels={
            "gpu": _make_vjp_kernel(interpret=False),
            "cpu": _make_vjp_kernel(interpret=True),
        },
        name="prolix.nonbonded_nl",
        context=context,
    )


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
    *,
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    compute_param_grads: bool | None = None,
    return_forces: bool = False,
    atom_perm: jax.Array | None = None,
    atom_inv_perm: jax.Array | None = None,
    nl_already_permuted: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Fused O(N*K) neighbor-list LJ + Coulomb/PME direct-space Pallas kernel (v2 O1, O2.1, O4, O5)."""
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
    q_arr = jnp.asarray(charges, dtype=dtype)
    sig_arr = jnp.asarray(sigmas, dtype=dtype)
    eps_arr = jnp.asarray(epsilons, dtype=dtype)
    nbr_arr = jnp.asarray(neighbor_idx, dtype=jnp.int32)

    inv_p = None
    if atom_perm is not None:
        perm = jnp.asarray(atom_perm, dtype=jnp.int32)
        inv_p = (
            jnp.argsort(perm).astype(jnp.int32)
            if atom_inv_perm is None
            else jnp.asarray(atom_inv_perm, dtype=jnp.int32)
        )
        positions = positions[perm]
        q_arr = q_arr[perm]
        sig_arr = sig_arr[perm]
        eps_arr = eps_arr[perm]
        msk_arr = msk_arr[perm]
        if not nl_already_permuted:
            nbr_arr = permute_neighbor_list_morton(nbr_arr, perm, inv_p, n_atoms=N)
            ei_arr, ev_arr, ee_arr = permute_and_sort_exclusions(
                ei_arr, ev_arr, ee_arr, perm, inv_p, n_atoms=N
            )

    if return_forces:
        if interpret is None:
            interpret = not _triton_gpu_available()
        e_val, f_val = _pallas_nonbonded_nl_vmappable(
            positions,
            q_arr,
            sig_arr,
            eps_arr,
            msk_arr,
            nbr_arr,
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
            str(excl_mode),
            float(scale_14_vdw),
            float(scale_14_elec),
        )
        if inv_p is not None:
            f_val = f_val[inv_p]
        return e_val, f_val

    if in_export_context():
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
        op = make_nonbonded_nl_op(
            has_box=bool(has_box),
            pme_alpha=float(pme_alpha),
            cutoff=float(cutoff),
            coulomb_constant=float(coulomb_constant),
            lj_switch_width=float(lj_switch_width),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            include_pme_excl_erf=bool(include_pme_excl_erf),
            excl_mode=str(excl_mode),
            scale_14_vdw=float(scale_14_vdw),
            scale_14_elec=float(scale_14_elec),
            compute_param_grads=compute_param_grads,
            context=ctx,
        )
        return op(
            positions, q_arr, sig_arr, eps_arr, msk_arr, nbr_arr, ei_arr, ev_arr, ee_arr, box_arr, lam_arr
        )

    if interpret is None:
        interpret = not _triton_gpu_available()

    # Use _pallas_nonbonded_nl_diff directly so custom_vjp(..., symbolic_zeros=True)
    # inspects arg.perturbed on each CustomVJPPrimal during standard MD grad(E)(pos)!
    return _pallas_nonbonded_nl_diff(
        positions,
        q_arr,
        sig_arr,
        eps_arr,
        msk_arr,
        nbr_arr,
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
        str(excl_mode),
        float(scale_14_vdw),
        float(scale_14_elec),
        compute_param_grads,
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
    orig_dtype = positions.dtype
    dtype = jnp.float32
    positions = positions.astype(dtype)
    charges = charges.astype(dtype)
    sigmas = sigmas.astype(dtype)
    epsilons = epsilons.astype(dtype)

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
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1).astype(orig_dtype)
    return jnp.sum(e_pad[:N]).astype(orig_dtype), forces


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


def flash_explicit_nonbonded_reference(
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
    *,
    has_box: bool = False,
    pme_alpha: float = 0.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
) -> jax.Array:
    """Plain-JAX reference for 2D-tiled FlashMD explicit nonbonded energy (2D or 3D batched)."""
    if positions.ndim == 3:
        return jax.vmap(
            lambda b: _export_flash_one(
                positions[b],
                charges[b] if charges.ndim == 2 else charges,
                sigmas[b] if sigmas.ndim == 2 else sigmas,
                epsilons[b] if epsilons.ndim == 2 else epsilons,
                atom_mask[b] if atom_mask.ndim == 2 else atom_mask,
                excl_indices[b] if excl_indices.ndim == 3 else excl_indices,
                excl_scales_vdw[b] if excl_scales_vdw.ndim == 3 else excl_scales_vdw,
                excl_scales_elec[b] if excl_scales_elec.ndim == 3 else excl_scales_elec,
                box[b] if box.ndim == 2 else box,
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
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        lam_arr,
        bool(has_box),
        float(pme_alpha),
        float(softcore_alpha),
        int(softcore_exponent),
    )


def make_flash_explicit_nonbonded_op(
    *,
    has_box: bool = False,
    pme_alpha: float = 0.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    context: Context | None = None,
):
    """Build an ``xtrax.op`` for 2D-tiled FlashMD explicit nonbonded energy."""
    hb = bool(has_box)
    pa = float(pme_alpha)
    sca = float(softcore_alpha)
    sce = int(softcore_exponent)
    ref = functools.partial(
        flash_explicit_nonbonded_reference,
        has_box=hb,
        pme_alpha=pa,
        softcore_alpha=sca,
        softcore_exponent=sce,
    )

    def _make_vjp_kernel(*, interpret: bool) -> VjpKernel:
        def _fwd(pos, q, sig, eps, msk, ei, ev, ee, bx, lam):
            if pos.ndim == 3:
                B = pos.shape[0]
                e, f = jax.vmap(
                    lambda b: _pallas_flash_explicit_vmappable(
                        pos[b],
                        q[b] if q.ndim == 2 else q,
                        sig[b] if sig.ndim == 2 else sig,
                        eps[b] if eps.ndim == 2 else eps,
                        msk[b] if msk.ndim == 2 else msk,
                        ei[b] if ei.ndim == 3 else ei,
                        ev[b] if ev.ndim == 3 else ev,
                        ee[b] if ee.ndim == 3 else ee,
                        bx[b] if bx.ndim == 2 else bx,
                        lam[b] if lam.ndim == 2 else lam,
                        hb,
                        pa,
                        sca,
                        sce,
                        interpret,
                    )
                )(jnp.arange(B))
                return e, (f, pos, q, sig, eps, msk, ei, ev, ee, bx, lam)
            return _pallas_flash_explicit_fwd(
                pos, q, sig, eps, msk, ei, ev, ee, bx, lam,
                hb, pa, sca, sce, interpret,
            )

        def _bwd(res, g):
            return _pallas_flash_explicit_bwd(
                hb, pa, sca, sce, interpret, res, g
            )

        return VjpKernel(fwd=_fwd, bwd=_bwd)

    return xtrax_op(
        ref,
        vjp_kernels={
            "gpu": _make_vjp_kernel(interpret=False),
            "cpu": _make_vjp_kernel(interpret=True),
        },
        name="prolix.flash_explicit_nonbonded",
        context=context,
    )


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
        ctx = Context(kernels=False, target="export", export_format="stablehlo")
    else:
        if interpret is None:
            interpret = not _triton_gpu_available()
        ctx = Context(kernels=True, backend="cpu" if interpret else "gpu")

    op = make_flash_explicit_nonbonded_op(
        has_box=bool(has_box),
        pme_alpha=float(pme_alpha),
        softcore_alpha=float(softcore_alpha),
        softcore_exponent=int(softcore_exponent),
        context=ctx,
    )
    return op(
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
    )


# =============================================================================
# MK-36: Single-Launch Fused Direct-Space Nonbonded NL + All-Bonded Megakernel
# =============================================================================

_MEGAKERNEL_BASE_NDIMS: tuple[int, ...] = (
    2,  # 0: positions
    1,  # 1: charges
    1,  # 2: sigmas
    1,  # 3: epsilons
    1,  # 4: atom_mask
    2,  # 5: neighbor_idx
    2,  # 6: excl_indices
    2,  # 7: excl_scales_vdw
    2,  # 8: excl_scales_elec
    2,  # 9: bond_indices
    2,  # 10: bond_params
    1,  # 11: bond_mask
    2,  # 12: angle_indices
    2,  # 13: angle_params
    1,  # 14: angle_mask
    2,  # 15: dih_indices
    2,  # 16: dih_params
    1,  # 17: dih_mask
    2,  # 18: harm_imp_indices
    2,  # 19: harm_imp_params
    1,  # 20: harm_imp_mask
    2,  # 21: exc14_pairs
    1,  # 22: exc14_sigmas
    1,  # 23: exc14_epsilons
    1,  # 24: exc14_chargeprods
    1,  # 25: exc14_mask
    2,  # 26: cmap_torsions
    1,  # 27: cmap_map_indices
    4,  # 28: cmap_coeffs
    1,  # 29: cmap_mask
    1,  # 30: box
    1,  # 31: lam_arr
)


def _pallas_bonded_and_nonbonded_nl_fused_batch(
    B: int,
    in_batched: tuple[bool, ...],
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
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
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    dist_eps: float = 1e-10,
    b0_is_ji: bool = True,
    angle_regularization_mode: str = "legacy_denom_eps",
    cmap_first_is_psi: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """2D-grid `(B, n_nb_blocks + total_bonded_blocks)` single-launch Pallas megakernel (MK-36)."""
    orig_dtype = positions.dtype
    dtype = jnp.float32 if (not interpret and orig_dtype == jnp.float64) else orig_dtype
    (
        b_pos,
        b_q,
        b_sig,
        b_eps,
        b_mask,
        b_nbr,
        b_ei,
        b_ev,
        b_ee,
        b_bi,
        b_bp,
        b_bm,
        b_ai,
        b_ap,
        b_am,
        b_di,
        b_dp,
        b_dm,
        b_hi,
        b_hp,
        b_hm,
        b_ep,
        b_es,
        b_ees,
        b_eq,
        b_em,
        b_ct,
        b_cmi,
        b_cc,
        b_cm,
        b_bx,
        b_lm,
    ) = in_batched

    N = int(positions.shape[1] if b_pos else positions.shape[0])
    K = int(neighbor_idx.shape[2] if b_nbr else neighbor_idx.shape[1])
    M = int(excl_indices.shape[2] if b_ei else excl_indices.shape[1])

    B_real = int(bond_indices.shape[1] if b_bi else bond_indices.shape[0])
    A_real = int(angle_indices.shape[1] if b_ai else angle_indices.shape[0])
    D_real = int(dih_indices.shape[1] if b_di else dih_indices.shape[0])
    H_real = int(harm_imp_indices.shape[1] if b_hi else harm_imp_indices.shape[0])
    E_real = int(exc14_pairs.shape[1] if b_ep else exc14_pairs.shape[0])
    C_real = int(cmap_torsions.shape[1] if b_ct else cmap_torsions.shape[0])

    if N == 0:
        return jnp.zeros((B,), dtype=orig_dtype), jnp.zeros((B, 0, 3), dtype=orig_dtype)

    n_nb_blocks = ((N + _NB_BLOCK_N - 1) // _NB_BLOCK_N) if K > 0 else 0
    n_blk_b = (B_real + _BLOCK_B - 1) // _BLOCK_B
    n_blk_a = (A_real + _BLOCK_B - 1) // _BLOCK_B
    n_blk_d = (D_real + _BLOCK_B - 1) // _BLOCK_B
    n_blk_h = (H_real + _BLOCK_B - 1) // _BLOCK_B
    n_blk_e = (E_real + _BLOCK_B - 1) // _BLOCK_B
    n_blk_c = ((C_real + _BLOCK_B - 1) // _BLOCK_B) if cmap_coeffs.size > 0 else 0

    o_b0, o_b1 = 0, n_blk_b
    o_a0, o_a1 = o_b1, o_b1 + n_blk_a
    o_d0, o_d1 = o_a1, o_a1 + n_blk_d
    o_h0, o_h1 = o_d1, o_d1 + n_blk_h
    o_e0, o_e1 = o_h1, o_h1 + n_blk_e
    o_c0, o_c1 = o_e1, o_e1 + n_blk_c
    total_bonded_blocks = o_c1
    total_blocks = n_nb_blocks + total_bonded_blocks

    if total_blocks == 0:
        return jnp.zeros((B,), dtype=orig_dtype), jnp.zeros((B, N, 3), dtype=orig_dtype)

    n_atom_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    N_pad = n_atom_blocks * _NB_BLOCK_N
    pad_n = N_pad - N
    k_blocks = max(1, (K + _NB_BLOCK_K - 1) // _NB_BLOCK_K)
    use_pre_encoded = excl_mode == "pre_encoded"

    pos_b = (positions if b_pos else positions[None]).astype(dtype)
    q_b = (charges if b_q else charges[None]).astype(dtype)
    sig_b = (sigmas if b_sig else sigmas[None]).astype(dtype)
    eps_b = (epsilons if b_eps else epsilons[None]).astype(dtype)
    mask_b = atom_mask if b_mask else atom_mask[None]
    nbr_b = neighbor_idx if b_nbr else neighbor_idx[None]
    ei_b = excl_indices if b_ei else excl_indices[None]
    ev_b = excl_scales_vdw if b_ev else excl_scales_vdw[None]
    ee_b = excl_scales_elec if b_ee else excl_scales_elec[None]
    bx_b = (box if b_bx else box[None]).astype(dtype)
    lam_b = (lam_arr if b_lm else lam_arr[None]).astype(dtype)

    B_pos = pos_b.shape[0]
    B_q = q_b.shape[0]
    B_sig = sig_b.shape[0]
    B_eps = eps_b.shape[0]
    B_mask = mask_b.shape[0]
    B_nbr = nbr_b.shape[0]
    B_ei = ei_b.shape[0]
    B_ev = ev_b.shape[0]
    B_ee = ee_b.shape[0]
    B_box = bx_b.shape[0]
    B_lam = lam_b.shape[0]

    if pad_n > 0:
        pos_pad = jnp.pad(pos_b, ((0, 0), (0, pad_n), (0, 0)))
        q_pad = jnp.pad(q_b, ((0, 0), (0, pad_n)))
        sig_pad = jnp.pad(sig_b, ((0, 0), (0, pad_n)), constant_values=1.0)
        eps_pad = jnp.pad(eps_b, ((0, 0), (0, pad_n)))
        mask_pad = jnp.pad(mask_b.astype(jnp.int32), ((0, 0), (0, pad_n)), constant_values=0)
    else:
        pos_pad = pos_b
        q_pad = q_b
        sig_pad = sig_b
        eps_pad = eps_b
        mask_pad = mask_b.astype(jnp.int32)

    if use_pre_encoded or n_nb_blocks == 0:
        excl_min_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_max_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_idx_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_vdw_flat = jnp.zeros((1,), dtype=dtype)
        excl_elec_flat = jnp.zeros((1,), dtype=dtype)
    else:
        pad_m = max(0, _NB_MAX_EXCL - M)
        pad_n_excl = N_pad - ei_b.shape[1]
        excl_idx_pad = ei_b[:, :, :_NB_MAX_EXCL].astype(jnp.int32)
        excl_vdw_pad = ev_b[:, :, :_NB_MAX_EXCL].astype(dtype)
        excl_elec_pad = ee_b[:, :, :_NB_MAX_EXCL].astype(dtype)
        if pad_n_excl > 0 or pad_m > 0:
            excl_idx_pad = jnp.pad(excl_idx_pad, ((0, 0), (0, pad_n_excl), (0, pad_m)), constant_values=-1)
            excl_vdw_pad = jnp.pad(excl_vdw_pad, ((0, 0), (0, pad_n_excl), (0, pad_m)), constant_values=1.0)
            excl_elec_pad = jnp.pad(excl_elec_pad, ((0, 0), (0, pad_n_excl), (0, pad_m)), constant_values=1.0)
        int_max_i32 = jnp.int32(0x7FFFFFFF)
        valid_e_pad = (excl_idx_pad >= 0) & (excl_idx_pad < N)
        excl_min_pad = jnp.min(jnp.where(valid_e_pad, excl_idx_pad, int_max_i32), axis=-1)
        excl_max_pad = jnp.max(jnp.where(valid_e_pad, excl_idx_pad, jnp.int32(-1)), axis=-1)
        excl_min_flat = jnp.asarray(excl_min_pad.reshape(B_ei * N_pad))
        excl_max_flat = jnp.asarray(excl_max_pad.reshape(B_ei * N_pad))
        excl_idx_flat = jnp.asarray(excl_idx_pad.reshape(B_ei * N_pad * _NB_MAX_EXCL))
        excl_vdw_flat = jnp.asarray(excl_vdw_pad.reshape(B_ev * N_pad * _NB_MAX_EXCL))
        excl_elec_flat = jnp.asarray(excl_elec_pad.reshape(B_ee * N_pad * _NB_MAX_EXCL))

    px = jnp.asarray(pos_pad[:, :, 0].reshape(B_pos * N_pad))
    py = jnp.asarray(pos_pad[:, :, 1].reshape(B_pos * N_pad))
    pz = jnp.asarray(pos_pad[:, :, 2].reshape(B_pos * N_pad))
    s_pos = N_pad if B_pos > 1 else 0

    q_flat = jnp.asarray(q_pad.reshape(B_q * N_pad))
    sig_flat = jnp.asarray(sig_pad.reshape(B_sig * N_pad))
    eps_flat = jnp.asarray(eps_pad.reshape(B_eps * N_pad))
    mask_flat = jnp.asarray(mask_pad.reshape(B_mask * N_pad))
    nbr_flat = (
        jnp.asarray(nbr_b.astype(jnp.int32).reshape(B_nbr * N * K))
        if K > 0
        else jnp.zeros((1,), dtype=jnp.int32)
    )
    box_len = int(bx_b.shape[-1])
    box_flat = jnp.asarray(bx_b.reshape(B_box * box_len))
    s_box = box_len if B_box > 1 else 0
    lam_flat = jnp.asarray(lam_b.reshape(B_lam))
    s_lam = 1 if B_lam > 1 else 0

    def _prep_int(
        arr: jax.Array, is_b: bool, real_n: int, pad_len: int, pad_val: int, col: int | None = None
    ) -> tuple[jax.Array, int]:
        stride = real_n + pad_len
        if real_n == 0:
            return jnp.full((stride,), pad_val, dtype=jnp.int32), 0
        x = arr[..., col].astype(jnp.int32) if col is not None else arr.astype(jnp.int32)
        if is_b:
            if pad_len > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_len)), constant_values=pad_val)
            return jnp.asarray(x.reshape(-1), dtype=jnp.int32), stride
        if pad_len > 0:
            x = jnp.pad(x, (0, pad_len), constant_values=pad_val)
        return jnp.asarray(x.reshape(-1), dtype=jnp.int32), 0

    def _prep_flt(
        arr: jax.Array, is_b: bool, real_n: int, pad_len: int, pad_val: float, col: int | None = None
    ) -> tuple[jax.Array, int]:
        stride = real_n + pad_len
        if real_n == 0:
            return jnp.full((stride,), pad_val, dtype=dtype), 0
        if col is not None and arr.shape[-1] <= col:
            return jnp.full((stride,), pad_val, dtype=dtype), 0
        x = arr[..., col].astype(dtype) if col is not None else arr.astype(dtype)
        if is_b:
            if pad_len > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_len)), constant_values=pad_val)
            return jnp.asarray(x.reshape(-1), dtype=dtype), stride
        if pad_len > 0:
            x = jnp.pad(x, (0, pad_len), constant_values=pad_val)
        return jnp.asarray(x.reshape(-1), dtype=dtype), 0

    B_pad = max(1, n_blk_b) * _BLOCK_B
    pad_b = B_pad - B_real
    b_i, s_bi = _prep_int(bond_indices, b_bi, B_real, pad_b, -1, 0)
    b_j, _ = _prep_int(bond_indices, b_bi, B_real, pad_b, -1, 1)
    b_r0, s_bp = _prep_flt(bond_params, b_bp, B_real, pad_b, 0.0, 0)
    b_k, _ = _prep_flt(bond_params, b_bp, B_real, pad_b, 0.0, 1)
    b_msk, s_bm = _prep_flt(bond_mask, b_bm, B_real, pad_b, 0.0, None)

    A_pad = max(1, n_blk_a) * _BLOCK_B
    pad_a = A_pad - A_real
    a_i, s_ai = _prep_int(angle_indices, b_ai, A_real, pad_a, -1, 0)
    a_j, _ = _prep_int(angle_indices, b_ai, A_real, pad_a, -1, 1)
    a_k, _ = _prep_int(angle_indices, b_ai, A_real, pad_a, -1, 2)
    a_th0, s_ap = _prep_flt(angle_params, b_ap, A_real, pad_a, 0.0, 0)
    a_kval, _ = _prep_flt(angle_params, b_ap, A_real, pad_a, 0.0, 1)
    a_msk, s_am = _prep_flt(angle_mask, b_am, A_real, pad_a, 0.0, None)

    D_pad = max(1, n_blk_d) * _BLOCK_B
    pad_d = D_pad - D_real
    d_i, s_di = _prep_int(dih_indices, b_di, D_real, pad_d, -1, 0)
    d_j, _ = _prep_int(dih_indices, b_di, D_real, pad_d, -1, 1)
    d_k, _ = _prep_int(dih_indices, b_di, D_real, pad_d, -1, 2)
    d_l, _ = _prep_int(dih_indices, b_di, D_real, pad_d, -1, 3)
    d_p0, s_dp = _prep_flt(dih_params, b_dp, D_real, pad_d, 0.0, 0)
    d_p1, _ = _prep_flt(dih_params, b_dp, D_real, pad_d, 0.0, 1)
    d_p2, _ = _prep_flt(dih_params, b_dp, D_real, pad_d, 0.0, 2)
    d_msk, s_dm = _prep_flt(dih_mask, b_dm, D_real, pad_d, 0.0, None)

    H_pad = max(1, n_blk_h) * _BLOCK_B
    pad_h = H_pad - H_real
    h_i, s_hi = _prep_int(harm_imp_indices, b_hi, H_real, pad_h, -1, 0)
    h_j, _ = _prep_int(harm_imp_indices, b_hi, H_real, pad_h, -1, 1)
    h_k, _ = _prep_int(harm_imp_indices, b_hi, H_real, pad_h, -1, 2)
    h_l, _ = _prep_int(harm_imp_indices, b_hi, H_real, pad_h, -1, 3)
    h_kval, s_hp = _prep_flt(harm_imp_params, b_hp, H_real, pad_h, 0.0, 0)
    h_phi0, _ = _prep_flt(harm_imp_params, b_hp, H_real, pad_h, 0.0, 1)
    h_msk, s_hm = _prep_flt(harm_imp_mask, b_hm, H_real, pad_h, 0.0, None)

    E_pad = max(1, n_blk_e) * _BLOCK_B
    pad_e = E_pad - E_real
    e_i, s_ep = _prep_int(exc14_pairs, b_ep, E_real, pad_e, -1, 0)
    e_j, _ = _prep_int(exc14_pairs, b_ep, E_real, pad_e, -1, 1)
    e_sig, s_es = _prep_flt(exc14_sigmas, b_es, E_real, pad_e, 1.0, None)
    e_eps, s_ee = _prep_flt(exc14_epsilons, b_ees, E_real, pad_e, 0.0, None)
    e_qp, s_eq = _prep_flt(exc14_chargeprods, b_eq, E_real, pad_e, 0.0, None)
    e_msk, s_em = _prep_flt(exc14_mask, b_em, E_real, pad_e, 0.0, None)

    C_pad = max(1, n_blk_c) * _BLOCK_B
    pad_c = C_pad - C_real
    c_0, s_ct = _prep_int(cmap_torsions, b_ct, C_real, pad_c, -1, 0)
    c_1, _ = _prep_int(cmap_torsions, b_ct, C_real, pad_c, -1, 1)
    c_2, _ = _prep_int(cmap_torsions, b_ct, C_real, pad_c, -1, 2)
    c_3, _ = _prep_int(cmap_torsions, b_ct, C_real, pad_c, -1, 3)
    c_4, _ = _prep_int(cmap_torsions, b_ct, C_real, pad_c, -1, 4)
    c_midx, s_cmi = _prep_int(cmap_map_indices, b_cmi, C_real, pad_c, 0, None)
    c_msk, s_cm = _prep_flt(cmap_mask, b_cm, C_real, pad_c, 0.0, None)

    cmap_base_ndim = 5 if b_cc else 4
    if cmap_coeffs.size > 0 and cmap_coeffs.ndim >= cmap_base_ndim:
        N_maps = int(cmap_coeffs.shape[1] if b_cc else cmap_coeffs.shape[0])
        G_cmap = int(cmap_coeffs.shape[2] if b_cc else cmap_coeffs.shape[1])
        c_coeffs_flat = jnp.asarray(cmap_coeffs.astype(dtype).reshape(-1))
        s_cc = (N_maps * G_cmap * G_cmap * 16) if b_cc else 0
    else:
        N_maps = 1
        G_cmap = 24
        c_coeffs_flat = jnp.zeros((16,), dtype=dtype)
        s_cc = 0

    e_atom_slots = n_nb_blocks * _NB_BLOCK_N
    e_out_dim = max(1, e_atom_slots + total_bonded_blocks)

    zero_fx = jnp.zeros((B * N_pad,), dtype=dtype)
    zero_fy = jnp.zeros((B * N_pad,), dtype=dtype)
    zero_fz = jnp.zeros((B * N_pad,), dtype=dtype)

    def _megakernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        sig_ref,
        eps_ref,
        mask_ref,
        nbr_flat_ref,
        excl_min_ref,
        excl_max_ref,
        excl_idx_flat_ref,
        excl_vdw_flat_ref,
        excl_elec_flat_ref,
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
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        pos_base = b_id * s_pos
        f_base = b_id * N_pad

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)
        six = jnp.asarray(6.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
        max_atom = max(N - 1, 0)

        def _load_box_b():
            base = b_id * s_box
            return tuple(
                _pl_load(box_ref, (base + jnp.full((1,), k, dtype=jnp.int32),))[0]
                for k in range(box_len)
            )

        def _do_nonbonded_nl(nb_pid):
            offs_i = nb_pid * _NB_BLOCK_N + jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
            in_bounds_i = offs_i < N
            safe_offs_i = jnp.minimum(offs_i, max_atom)
            e_nb_base = b_id * e_out_dim

            base_q = (b_id * N_pad) if B_q > 1 else jnp.int32(0)
            base_sig = (b_id * N_pad) if B_sig > 1 else jnp.int32(0)
            base_eps = (b_id * N_pad) if B_eps > 1 else jnp.int32(0)
            base_mask = (b_id * N_pad) if B_mask > 1 else jnp.int32(0)
            base_nbr = (b_id * (N * K)) if B_nbr > 1 else jnp.int32(0)
            base_box = b_id * s_box
            base_lam = b_id * s_lam

            mask_i = (_pl_load(mask_ref, (base_mask + offs_i,)) > 0) & in_bounds_i
            px_i = _pl_load(px_ref, (pos_base + offs_i,))
            py_i = _pl_load(py_ref, (pos_base + offs_i,))
            pz_i = _pl_load(pz_ref, (pos_base + offs_i,))
            q_i = _pl_load(q_ref, (base_q + offs_i,))
            sig_i = _pl_load(sig_ref, (base_sig + offs_i,))
            eps_i = _pl_load(eps_ref, (base_eps + offs_i,))

            if not use_pre_encoded:
                base_ei = (b_id * N_pad) if B_ei > 1 else jnp.int32(0)
                base_ei_flat = (b_id * (N_pad * _NB_MAX_EXCL)) if B_ei > 1 else jnp.int32(0)
                base_ev_flat = (b_id * (N_pad * _NB_MAX_EXCL)) if B_ev > 1 else jnp.int32(0)
                base_ee_flat = (b_id * (N_pad * _NB_MAX_EXCL)) if B_ee > 1 else jnp.int32(0)
                excl_min_i = _pl_load(excl_min_ref, (base_ei + offs_i,))
                excl_max_i = _pl_load(excl_max_ref, (base_ei + offs_i,))
                offs_m = jnp.arange(_NB_MAX_EXCL, dtype=jnp.int32)
                flat_excl_offs = (offs_i[:, None] * _NB_MAX_EXCL + offs_m[None, :]).reshape(
                    _NB_BLOCK_N * _NB_MAX_EXCL
                )

            ax = _pl_load(box_ref, (base_box + jnp.zeros((1,), dtype=jnp.int32),))[0]
            by = _pl_load(box_ref, (base_box + jnp.ones((1,), dtype=jnp.int32),))[0]
            cz = _pl_load(box_ref, (base_box + jnp.full((1,), 2, dtype=jnp.int32),))[0]
            bx_t = _pl_load(box_ref, (base_box + jnp.full((1,), 3, dtype=jnp.int32),))[0]
            cx_t = _pl_load(box_ref, (base_box + jnp.full((1,), 4, dtype=jnp.int32),))[0]
            cy_t = _pl_load(box_ref, (base_box + jnp.full((1,), 5, dtype=jnp.int32),))[0]

            s14_vdw = jnp.asarray(scale_14_vdw, dtype=dtype)
            s14_elec = jnp.asarray(scale_14_elec, dtype=dtype)
            if has_box:
                pos_big = jnp.asarray(1e30, dtype=dtype)
                neg_big = jnp.asarray(-1e30, dtype=dtype)
                r_min_x = jnp.min(jnp.where(mask_i, px_i, pos_big))
                r_max_x = jnp.max(jnp.where(mask_i, px_i, neg_big))
                r_min_y = jnp.min(jnp.where(mask_i, py_i, pos_big))
                r_max_y = jnp.max(jnp.where(mask_i, py_i, neg_big))
                r_min_z = jnp.min(jnp.where(mask_i, pz_i, pos_big))
                r_max_z = jnp.max(jnp.where(mask_i, pz_i, neg_big))
                is_ortho = (bx_t == zero) & (cx_t == zero) & (cy_t == zero)
                r_m_val = jnp.asarray(float(max(cutoff, 6.0)), dtype=dtype)
                interior_x = is_ortho & (r_min_x >= r_m_val) & (r_max_x <= ax - r_m_val)
                interior_y = is_ortho & (r_min_y >= r_m_val) & (r_max_y <= by - r_m_val)
                interior_z = is_ortho & (r_min_z >= r_m_val) & (r_max_z <= cz - r_m_val)
                interior_all = interior_x & interior_y & interior_z
            else:
                interior_all = jnp.bool_(True)

            lam = _pl_load(lam_ref, (base_lam + jnp.zeros((1,), dtype=jnp.int32),))[0]
            one_minus_lam = one - lam
            soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * (
                one_minus_lam ** int(softcore_exponent)
            )
            coul_const = jnp.asarray(coulomb_constant, dtype=dtype)
            alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
            two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

            def _k_tile_body(k_tile, carry):
                offs_k = k_tile * _NB_BLOCK_K + jnp.arange(_NB_BLOCK_K, dtype=jnp.int32)
                in_bounds_nk = in_bounds_i[:, None] & (offs_k[None, :] < K)
                safe_offs_k = jnp.minimum(offs_k, max(K - 1, 0))
                flat_nbr_offs = base_nbr + (safe_offs_i[:, None] * K + safe_offs_k[None, :]).reshape(
                    _NB_BLOCK_N * _NB_BLOCK_K
                )
                raw_j_1d = _pl_load(nbr_flat_ref, (flat_nbr_offs,))
                raw_j = jnp.where(in_bounds_nk, raw_j_1d.reshape(_NB_BLOCK_N, _NB_BLOCK_K), jnp.int32(N))
                is_full_excl = (raw_j >= 0) & ((raw_j & jnp.int32(_NB_EXCL_FULL_BIT)) != 0)
                is_14_excl = (raw_j >= 0) & ((raw_j & jnp.int32(_NB_EXCL_14_BIT)) != 0)
                j_idx = jnp.where(raw_j >= 0, raw_j & jnp.int32(_NB_INDEX_MASK), raw_j)
                cand_pair = (
                    mask_i[:, None]
                    & (j_idx >= 0)
                    & (j_idx < N)
                    & (j_idx != offs_i[:, None])
                    & (True if (not use_pre_encoded or include_pme_excl_erf) else (~is_full_excl))
                )
                has_any_valid = jnp.max(cand_pair.astype(jnp.int32)) > 0

                def _eval_nonempty_tile(c_in):
                    acc_e, acc_fx, acc_fy, acc_fz = c_in
                    safe_j_1d = jnp.clip(j_idx.reshape(_NB_BLOCK_N * _NB_BLOCK_K), 0, N_pad - 1)

                    px_j = _pl_load(px_ref, (pos_base + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                    py_j = _pl_load(py_ref, (pos_base + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                    pz_j = _pl_load(pz_ref, (pos_base + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                    q_j = _pl_load(q_ref, (base_q + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                    sig_j = _pl_load(sig_ref, (base_sig + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                    eps_j = _pl_load(eps_ref, (base_eps + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K)
                    mask_j = _pl_load(mask_ref, (base_mask + safe_j_1d,)).reshape(_NB_BLOCK_N, _NB_BLOCK_K) > 0

                    valid_pair = cand_pair & mask_j

                    raw_dx = px_i[:, None] - px_j
                    raw_dy = py_i[:, None] - py_j
                    raw_dz = pz_i[:, None] - pz_j
                    if has_box:
                        dx, dy, dz = jax.lax.cond(
                            interior_all,
                            lambda d: d,
                            lambda d: _min_image_triclinic_xyz(
                                d[0], d[1], d[2], ax, by, cz, bx_t, cx_t, cy_t, True
                            ),
                            (raw_dx, raw_dy, raw_dz),
                        )
                    else:
                        dx, dy, dz = raw_dx, raw_dy, raw_dz

                    r_sq = dx * dx + dy * dy + dz * dz + jnp.asarray(1e-12, dtype=dtype)
                    dist = jnp.sqrt(r_sq)
                    in_cutoff = dist < jnp.asarray(cutoff, dtype=dtype)
                    active = valid_pair & in_cutoff

                    if use_pre_encoded:
                        w_vdw = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, one))
                        w_elec = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_elec, one))
                    else:
                        has_tag = is_full_excl | is_14_excl
                        in_excl_range = (
                            active
                            & (~has_tag)
                            & (j_idx >= excl_min_i[:, None])
                            & (j_idx <= excl_max_i[:, None])
                        )

                        def _lookup_excl_tile(_):
                            excl_idx_i = _pl_load(excl_idx_flat_ref, (base_ei_flat + flat_excl_offs,)).reshape(
                                _NB_BLOCK_N, _NB_MAX_EXCL
                            )
                            excl_vdw_i = _pl_load(excl_vdw_flat_ref, (base_ev_flat + flat_excl_offs,)).reshape(
                                _NB_BLOCK_N, _NB_MAX_EXCL
                            )
                            excl_elec_i = _pl_load(excl_elec_flat_ref, (base_ee_flat + flat_excl_offs,)).reshape(
                                _NB_BLOCK_N, _NB_MAX_EXCL
                            )
                            valid_excl_i = (excl_idx_i >= 0) & (excl_idx_i < N)
                            matches = (j_idx[:, :, None] == excl_idx_i[:, None, :]) & valid_excl_i[:, None, :]
                            any_match = jnp.sum(matches.astype(jnp.int32), axis=2) > 0
                            wv = jnp.where(
                                any_match,
                                jnp.sum(jnp.where(matches, excl_vdw_i[:, None, :], zero), axis=2),
                                one,
                            )
                            we = jnp.where(
                                any_match,
                                jnp.sum(jnp.where(matches, excl_elec_i[:, None, :], zero), axis=2),
                                one,
                            )
                            return wv, we

                        def _no_excl_tile(_):
                            ones_t = jnp.ones((_NB_BLOCK_N, _NB_BLOCK_K), dtype=dtype)
                            return ones_t, ones_t

                        w_vdw_tbl, w_elec_tbl = jax.lax.cond(
                            jnp.sum(in_excl_range.astype(jnp.int32)) > 0,
                            _lookup_excl_tile,
                            _no_excl_tile,
                            None,
                        )
                        w_vdw = jnp.where(
                            is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, w_vdw_tbl)
                        )
                        w_elec = jnp.where(
                            is_full_excl, zero, jnp.where(is_14_excl, s14_elec, w_elec_tbl)
                        )

                    sig_ij = half * (sig_i[:, None] + sig_j)
                    sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
                    eps_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j, zero) + jnp.asarray(1e-30, dtype=dtype))

                    u = dist / sig_safe
                    u6 = u ** 6
                    denom_lj = soft_alpha_lj + u6
                    x_inv6 = one / jnp.maximum(denom_lj, jnp.asarray(1e-6, dtype=dtype))
                    v_unsw = four * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)
                    dv_unsw_dr = (
                        jnp.asarray(-24.0, dtype=dtype)
                        * eps_ij
                        * lam
                        * (two * x_inv6 - one)
                        * (x_inv6 * x_inv6)
                        * u6
                        / dist
                    )

                    s_w, ds_dr = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
                    e_lj = w_vdw * v_unsw * s_w
                    de_lj_dr = w_vdw * (dv_unsw_dr * s_w + v_unsw * ds_dr)

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
                    f_scalar = jnp.where(active, (zero - (de_lj_dr + de_coul_dr)) / dist, zero)

                    acc_e = acc_e + half * jnp.sum(e_pair, axis=1)
                    acc_fx = acc_fx + jnp.sum(f_scalar * dx, axis=1)
                    acc_fy = acc_fy + jnp.sum(f_scalar * dy, axis=1)
                    acc_fz = acc_fz + jnp.sum(f_scalar * dz, axis=1)
                    return (acc_e, acc_fx, acc_fy, acc_fz)

                return jax.lax.cond(has_any_valid, _eval_nonempty_tile, lambda c_in: c_in, carry)

            def _run_k_tiles(c0):
                e_blk, fx_blk, fy_blk, fz_blk = jax.lax.fori_loop(0, k_blocks, _k_tile_body, c0)
                if total_bonded_blocks == 0:
                    _pl_store(fx_out_ref, (f_base + offs_i,), fx_blk, mask=mask_i)
                    _pl_store(fy_out_ref, (f_base + offs_i,), fy_blk, mask=mask_i)
                    _pl_store(fz_out_ref, (f_base + offs_i,), fz_blk, mask=mask_i)
                else:
                    _pl_atomic_add_1d(fx_out_ref, f_base + offs_i, fx_blk, mask_i, interpret)
                    _pl_atomic_add_1d(fy_out_ref, f_base + offs_i, fy_blk, mask_i, interpret)
                    _pl_atomic_add_1d(fz_out_ref, f_base + offs_i, fz_blk, mask_i, interpret)
                _pl_store(e_out_ref, (e_nb_base + offs_i,), jnp.where(mask_i, e_blk, zero))
                return nb_pid - nb_pid

            def _skip_k_tiles(_c0):
                _pl_store(e_out_ref, (e_nb_base + offs_i,), (offs_i - offs_i).astype(dtype))
                return nb_pid - nb_pid

            init_carry = (
                jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
                jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
                jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
                jnp.zeros((_NB_BLOCK_N,), dtype=dtype),
            )
            return jax.lax.cond(
                jnp.max(mask_i.astype(jnp.int32)) > 0,
                _run_k_tiles,
                _skip_k_tiles,
                init_carry,
            )

        def _do_bonds(p_val):
            offs = (p_val - o_b0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            mv = _pl_load(b_msk_ref, (b_id * s_bm + offs,))
            has_any = jnp.max(((offs < B_real) & (mv > zero)).astype(jnp.int32)) > 0
            zero_e_val = (p_val - p_val).astype(dtype)

            def _eval_bonds(_):
                ai = _pl_load(b_i_ref, (b_id * s_bi + offs,))
                aj = _pl_load(b_j_ref, (b_id * s_bi + offs,))
                valid = (offs < B_real) & (mv > zero) & (ai >= 0) & (ai < N) & (aj >= 0) & (aj < N)
                m_eff = jnp.where(valid, mv, zero)
                si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
                sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
                ri = (_pl_load(px_ref, (pos_base + si,)), _pl_load(py_ref, (pos_base + si,)), _pl_load(pz_ref, (pos_base + si,)))
                rj = (_pl_load(px_ref, (pos_base + sj,)), _pl_load(py_ref, (pos_base + sj,)), _pl_load(pz_ref, (pos_base + sj,)))
                d_vec = _min_image3(_sub3(ri, rj), _load_box_b(), has_box)
                d_sq = _dot3(d_vec, d_vec)
                if dist_eps > 0.0:
                    dist = jnp.sqrt(d_sq + jnp.asarray(dist_eps, dtype=dtype))
                else:
                    d_sq_s = jnp.where(d_sq > zero, d_sq, one)
                    dist = jnp.where(d_sq > zero, jnp.sqrt(d_sq_s), zero)
                r0_b = _pl_load(b_r0_ref, (b_id * s_bp + offs,))
                k_b = _pl_load(b_k_ref, (b_id * s_bp + offs,))
                diff = dist - r0_b
                e_b = half * m_eff * k_b * diff * diff
                dist_denom = jnp.where(dist > zero, dist, one)
                f_coeff = jnp.where(valid & (dist > zero), (zero - m_eff * k_b * diff) / dist_denom, zero)
                fi = _scale3(f_coeff, d_vec)
                fj = _scale3(zero - one, fi)
                _pl_atomic_add_1d(fx_out_ref, f_base + si, fi[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + si, fi[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + si, fi[2], valid, interpret)
                _pl_atomic_add_1d(fx_out_ref, f_base + sj, fj[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + sj, fj[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + sj, fj[2], valid, interpret)
                return jnp.sum(e_b)

            return jax.lax.cond(has_any, _eval_bonds, lambda z: z, zero_e_val)

        def _do_angles(p_val):
            offs = (p_val - o_a0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            mv = _pl_load(a_msk_ref, (b_id * s_am + offs,))
            has_any = jnp.max(((offs < A_real) & (mv > zero)).astype(jnp.int32)) > 0
            zero_e_val = (p_val - p_val).astype(dtype)

            def _eval_angles(_):
                ai = _pl_load(a_i_ref, (b_id * s_ai + offs,))
                aj = _pl_load(a_j_ref, (b_id * s_ai + offs,))
                ak = _pl_load(a_k_ref, (b_id * s_ai + offs,))
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
                ri = (_pl_load(px_ref, (pos_base + si,)), _pl_load(py_ref, (pos_base + si,)), _pl_load(pz_ref, (pos_base + si,)))
                rj = (_pl_load(px_ref, (pos_base + sj,)), _pl_load(py_ref, (pos_base + sj,)), _pl_load(pz_ref, (pos_base + sj,)))
                rk = (_pl_load(px_ref, (pos_base + sk,)), _pl_load(py_ref, (pos_base + sk,)), _pl_load(pz_ref, (pos_base + sk,)))
                box_xyz = _load_box_b()
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
                th0 = _pl_load(a_th0_ref, (b_id * s_ap + offs,))
                k_a = _pl_load(a_kval_ref, (b_id * s_ap + offs,))
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
                _pl_atomic_add_1d(fx_out_ref, f_base + si, fi[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + si, fi[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + si, fi[2], valid, interpret)
                _pl_atomic_add_1d(fx_out_ref, f_base + sj, fj[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + sj, fj[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + sj, fj[2], valid, interpret)
                _pl_atomic_add_1d(fx_out_ref, f_base + sk, fk[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + sk, fk[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + sk, fk[2], valid, interpret)
                return jnp.sum(e_a)

            return jax.lax.cond(has_any, _eval_angles, lambda z: z, zero_e_val)

        def _do_torsions_periodic(p_val):
            offs = (p_val - o_d0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            mv = _pl_load(d_msk_ref, (b_id * s_dm + offs,))
            has_any = jnp.max(((offs < D_real) & (mv > zero)).astype(jnp.int32)) > 0
            zero_e_val = (p_val - p_val).astype(dtype)

            def _eval_torsions(_):
                ai = _pl_load(d_i_ref, (b_id * s_di + offs,))
                aj = _pl_load(d_j_ref, (b_id * s_di + offs,))
                ak = _pl_load(d_k_ref, (b_id * s_di + offs,))
                al = _pl_load(d_l_ref, (b_id * s_di + offs,))
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
                ri = (_pl_load(px_ref, (pos_base + si,)), _pl_load(py_ref, (pos_base + si,)), _pl_load(pz_ref, (pos_base + si,)))
                rj = (_pl_load(px_ref, (pos_base + sj,)), _pl_load(py_ref, (pos_base + sj,)), _pl_load(pz_ref, (pos_base + sj,)))
                rk = (_pl_load(px_ref, (pos_base + sk,)), _pl_load(py_ref, (pos_base + sk,)), _pl_load(pz_ref, (pos_base + sk,)))
                rl = (_pl_load(px_ref, (pos_base + sl,)), _pl_load(py_ref, (pos_base + sl,)), _pl_load(pz_ref, (pos_base + sl,)))
                phi, dphi_dri, dphi_drj, dphi_drk, dphi_drl = _torsion_angle_and_atom_grads_vec3(
                    ri, rj, rk, rl, valid, _load_box_b(), has_box, b0_is_ji
                )
                prm0 = _pl_load(d_p0_ref, (b_id * s_dp + offs,))
                prm1 = _pl_load(d_p1_ref, (b_id * s_dp + offs,))
                prm2 = _pl_load(d_p2_ref, (b_id * s_dp + offs,))
                arg = prm0 * phi - prm1
                e_d = m_eff * prm2 * (one + jnp.cos(arg))
                g_phi = (zero - m_eff) * prm2 * prm0 * jnp.sin(arg)
                neg_g_phi = jnp.where(valid, zero - g_phi, zero)
                fi = _scale3(neg_g_phi, dphi_dri)
                fj = _scale3(neg_g_phi, dphi_drj)
                fk = _scale3(neg_g_phi, dphi_drk)
                fl = _scale3(neg_g_phi, dphi_drl)
                for s_atom, f_atom in ((si, fi), (sj, fj), (sk, fk), (sl, fl)):
                    _pl_atomic_add_1d(fx_out_ref, f_base + s_atom, f_atom[0], valid, interpret)
                    _pl_atomic_add_1d(fy_out_ref, f_base + s_atom, f_atom[1], valid, interpret)
                    _pl_atomic_add_1d(fz_out_ref, f_base + s_atom, f_atom[2], valid, interpret)
                return jnp.sum(e_d)

            return jax.lax.cond(has_any, _eval_torsions, lambda z: z, zero_e_val)

        def _do_impropers_harmonic(p_val):
            offs = (p_val - o_h0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            mv = _pl_load(h_msk_ref, (b_id * s_hm + offs,))
            has_any = jnp.max(((offs < H_real) & (mv > zero)).astype(jnp.int32)) > 0
            zero_e_val = (p_val - p_val).astype(dtype)

            def _eval_impropers(_):
                ai = _pl_load(h_i_ref, (b_id * s_hi + offs,))
                aj = _pl_load(h_j_ref, (b_id * s_hi + offs,))
                ak = _pl_load(h_k_ref, (b_id * s_hi + offs,))
                al = _pl_load(h_l_ref, (b_id * s_hi + offs,))
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
                ri = (_pl_load(px_ref, (pos_base + si,)), _pl_load(py_ref, (pos_base + si,)), _pl_load(pz_ref, (pos_base + si,)))
                rj = (_pl_load(px_ref, (pos_base + sj,)), _pl_load(py_ref, (pos_base + sj,)), _pl_load(pz_ref, (pos_base + sj,)))
                rk = (_pl_load(px_ref, (pos_base + sk,)), _pl_load(py_ref, (pos_base + sk,)), _pl_load(pz_ref, (pos_base + sk,)))
                rl = (_pl_load(px_ref, (pos_base + sl,)), _pl_load(py_ref, (pos_base + sl,)), _pl_load(pz_ref, (pos_base + sl,)))
                phi, dphi_dri, dphi_drj, dphi_drk, dphi_drl = _torsion_angle_and_atom_grads_vec3(
                    ri, rj, rk, rl, valid, _load_box_b(), has_box, b0_is_ji
                )
                k_h = _pl_load(h_kval_ref, (b_id * s_hp + offs,))
                phi0_h = _pl_load(h_phi0_ref, (b_id * s_hp + offs,))
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
                    _pl_atomic_add_1d(fx_out_ref, f_base + s_atom, f_atom[0], valid, interpret)
                    _pl_atomic_add_1d(fy_out_ref, f_base + s_atom, f_atom[1], valid, interpret)
                    _pl_atomic_add_1d(fz_out_ref, f_base + s_atom, f_atom[2], valid, interpret)
                return jnp.sum(e_h)

            return jax.lax.cond(has_any, _eval_impropers, lambda z: z, zero_e_val)

        def _do_exceptions_14(p_val):
            offs = (p_val - o_e0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            mv = _pl_load(e_msk_ref, (b_id * s_em + offs,))
            has_any = jnp.max(((offs < E_real) & (mv > zero)).astype(jnp.int32)) > 0
            zero_e_val = (p_val - p_val).astype(dtype)

            def _eval_exceptions(_):
                coul = jnp.asarray(coulomb_constant, dtype=dtype)
                ai = _pl_load(e_i_ref, (b_id * s_ep + offs,))
                aj = _pl_load(e_j_ref, (b_id * s_ep + offs,))
                valid = (offs < E_real) & (mv > zero) & (ai >= 0) & (ai < N) & (aj >= 0) & (aj < N) & (ai != aj)
                m_eff = jnp.where(valid, mv, zero)
                si = jnp.where(valid, jnp.clip(ai, 0, max_atom), 0)
                sj = jnp.where(valid, jnp.clip(aj, 0, max_atom), 0)
                ri = (_pl_load(px_ref, (pos_base + si,)), _pl_load(py_ref, (pos_base + si,)), _pl_load(pz_ref, (pos_base + si,)))
                rj = (_pl_load(px_ref, (pos_base + sj,)), _pl_load(py_ref, (pos_base + sj,)), _pl_load(pz_ref, (pos_base + sj,)))
                lam_v = _pl_load(lam_ref, (b_id * s_lam + jnp.zeros((1,), dtype=jnp.int32),))[0]
                d_vec = _min_image3(_sub3(ri, rj), _load_box_b(), has_box)
                r_sq = _dot3(d_vec, d_vec) + jnp.asarray(1e-12, dtype=dtype)
                dist = jnp.sqrt(r_sq)
                sig_e = _pl_load(e_sig_ref, (b_id * s_es + offs,))
                eps_e = _pl_load(e_eps_ref, (b_id * s_ee + offs,))
                qp_e = _pl_load(e_qp_ref, (b_id * s_eq + offs,))
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
                _pl_atomic_add_1d(fx_out_ref, f_base + si, fi[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + si, fi[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + si, fi[2], valid, interpret)
                _pl_atomic_add_1d(fx_out_ref, f_base + sj, fj[0], valid, interpret)
                _pl_atomic_add_1d(fy_out_ref, f_base + sj, fj[1], valid, interpret)
                _pl_atomic_add_1d(fz_out_ref, f_base + sj, fj[2], valid, interpret)
                return jnp.sum(e_pair)

            return jax.lax.cond(has_any, _eval_exceptions, lambda z: z, zero_e_val)

        def _do_cmap(p_val):
            offs = (p_val - o_c0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            mv = _pl_load(c_msk_ref, (b_id * s_cm + offs,))
            has_any = jnp.max(((offs < C_real) & (mv > zero)).astype(jnp.int32)) > 0
            zero_e_val = (p_val - p_val).astype(dtype)

            def _eval_cmap(_):
                delta = jnp.asarray(2.0 * jnp.pi / G_cmap, dtype=dtype)
                inv_delta = one / delta
                i0 = _pl_load(c_0_ref, (b_id * s_ct + offs,))
                i1 = _pl_load(c_1_ref, (b_id * s_ct + offs,))
                i2 = _pl_load(c_2_ref, (b_id * s_ct + offs,))
                i3 = _pl_load(c_3_ref, (b_id * s_ct + offs,))
                i4 = _pl_load(c_4_ref, (b_id * s_ct + offs,))
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
                r0 = (_pl_load(px_ref, (pos_base + s0,)), _pl_load(py_ref, (pos_base + s0,)), _pl_load(pz_ref, (pos_base + s0,)))
                r1 = (_pl_load(px_ref, (pos_base + s1,)), _pl_load(py_ref, (pos_base + s1,)), _pl_load(pz_ref, (pos_base + s1,)))
                r2 = (_pl_load(px_ref, (pos_base + s2,)), _pl_load(py_ref, (pos_base + s2,)), _pl_load(pz_ref, (pos_base + s2,)))
                r3 = (_pl_load(px_ref, (pos_base + s3,)), _pl_load(py_ref, (pos_base + s3,)), _pl_load(pz_ref, (pos_base + s3,)))
                r4 = (_pl_load(px_ref, (pos_base + s4,)), _pl_load(py_ref, (pos_base + s4,)), _pl_load(pz_ref, (pos_base + s4,)))
                box_xyz = _load_box_b()
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
                m_raw = _pl_load(c_midx_ref, (b_id * s_cmi + offs,))
                m_safe = jnp.where(valid, jnp.clip(m_raw, 0, max(N_maps - 1, 0)), 0)
                base_idx = (b_id * s_cc) + ((m_safe * G_cmap + g1) * G_cmap + g2) * 16
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
                    _pl_atomic_add_1d(fx_out_ref, f_base + s_atom, f_atom[0], valid, interpret)
                    _pl_atomic_add_1d(fy_out_ref, f_base + s_atom, f_atom[1], valid, interpret)
                    _pl_atomic_add_1d(fz_out_ref, f_base + s_atom, f_atom[2], valid, interpret)
                return jnp.sum(e_c)

            return jax.lax.cond(has_any, _eval_cmap, lambda z: z, zero_e_val)

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

        def _do_bonded_all(b_pid):
            e_sum = _dispatch_group(0, b_pid)
            _pl_store(
                e_out_ref,
                (jnp.full((1,), b_id * e_out_dim + e_atom_slots + b_pid, dtype=jnp.int32),),
                e_sum[None],
            )
            return b_pid - b_pid

        if n_nb_blocks > 0 and total_bonded_blocks > 0:
            _ = jax.lax.cond(
                pid < n_nb_blocks,
                _do_nonbonded_nl,
                lambda p: _do_bonded_all(p - n_nb_blocks),
                pid,
            )
        elif n_nb_blocks > 0:
            _ = _do_nonbonded_nl(pid)
        else:
            _ = _do_bonded_all(pid)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(58)]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes = [
        jax.ShapeDtypeStruct((B * e_out_dim,), dtype),
        jax.ShapeDtypeStruct((B * N_pad,), dtype),
        jax.ShapeDtypeStruct((B * N_pad,), dtype),
        jax.ShapeDtypeStruct((B * N_pad,), dtype),
    ]
    e_blk, fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _megakernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(B, total_blocks),
        input_output_aliases={55: 1, 56: 2, 57: 3},
        interpret=interpret,
        compiler_params=(
            resolve_pallas_compiler_params("megakernel", grid_blocks=B * total_blocks)
            if not interpret
            else None
        ),
        name="pallas_bonded_and_nonbonded_nl_fused",
    )(
        px,
        py,
        pz,
        q_flat,
        sig_flat,
        eps_flat,
        mask_flat,
        nbr_flat,
        excl_min_flat,
        excl_max_flat,
        excl_idx_flat,
        excl_vdw_flat,
        excl_elec_flat,
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
        box_flat,
        lam_flat,
        zero_fx,
        zero_fy,
        zero_fz,
    )
    fx_b = fx_pad.reshape((B, N_pad))[:, :N]
    fy_b = fy_pad.reshape((B, N_pad))[:, :N]
    fz_b = fz_pad.reshape((B, N_pad))[:, :N]
    forces = jnp.stack([fx_b, fy_b, fz_b], axis=-1).astype(orig_dtype)
    energies = jnp.sum(e_blk.reshape((B, e_out_dim)).astype(orig_dtype), axis=-1)
    return energies, forces


def _pallas_bonded_and_nonbonded_nl_fused_single(
    *args,
) -> tuple[jax.Array, jax.Array]:
    arr_args = args[:32]
    static_args = args[32:]
    e_b, f_b = _pallas_bonded_and_nonbonded_nl_fused_batch(
        1,
        (False,) * 32,
        *arr_args,
        *static_args,
    )
    return e_b[0], f_b[0]


def _pallas_bonded_and_nonbonded_nl_fused_vmappable(
    *args,
) -> tuple[jax.Array, jax.Array]:
    arr_args = args[:32]
    static_args = args[32:]

    @jax.custom_batching.custom_vmap
    def _vmappable(*a):
        return _pallas_bonded_and_nonbonded_nl_fused_single(*a, *static_args)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, *a):
        nbr = a[5]
        n_nbr = int(nbr.shape[-2]) * int(nbr.shape[-1])
        b_nbr = int(axis_size) if in_batched[5] else 1
        if b_nbr * n_nbr < (1 << 30):
            e_batch, f_batch = _pallas_bonded_and_nonbonded_nl_fused_batch(
                int(axis_size),
                tuple(bool(ib) for ib in in_batched),
                *a,
                *static_args,
            )
            return (e_batch, f_batch), (True, True)

        def _step(b_idx):
            step_in = [x[b_idx] if ib else x for x, ib in zip(a, in_batched)]
            return _vmappable(*step_in)

        e_batch, f_batch = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_batch, f_batch), (True, True)

    positions = arr_args[0]
    if positions.ndim == 3:
        B = int(positions.shape[0])
        in_b = tuple(bool(a.ndim > bn) for a, bn in zip(arr_args, _MEGAKERNEL_BASE_NDIMS))
        nbr = arr_args[5]
        n_nbr = int(nbr.shape[-2]) * int(nbr.shape[-1])
        b_nbr = B if in_b[5] else 1
        if b_nbr * n_nbr < (1 << 30):
            return _pallas_bonded_and_nonbonded_nl_fused_batch(
                B,
                in_b,
                *arr_args,
                *static_args,
            )
        return jax.vmap(
            lambda b: _vmappable(*[x[b] if ib else x for x, ib in zip(arr_args, in_b)])
        )(jnp.arange(B))

    return _vmappable(*arr_args)


@functools.partial(
    custom_vjp,
    nondiff_argnums=(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48),
)
def _pallas_bonded_and_nonbonded_nl_fused_diff(*args):
    energy, _ = _pallas_bonded_and_nonbonded_nl_fused_vmappable(*args[:-1])
    return energy


def _pallas_bonded_and_nonbonded_nl_fused_fwd(*args):
    raw_arrs = args[:32]
    static_args = args[32:-1]
    compute_param_grads = args[-1]
    vals = tuple(_unwrap_primal(a) for a in raw_arrs)
    energy, forces = _pallas_bonded_and_nonbonded_nl_fused_vmappable(*vals, *static_args)

    default_pert = True if compute_param_grads is None else bool(compute_param_grads)
    if compute_param_grads is not None:
        any_param_perturbed = bool(compute_param_grads)
    else:
        any_param_perturbed = any(
            _is_perturbed(raw_arrs[idx], default_pert)
            for idx in (1, 2, 3, 7, 8, 10, 13, 16, 19, 22, 23, 24, 28, 31)
        )
    shapes_meta = _make_shapes_meta(*vals)
    param_res = vals if any_param_perturbed else None
    return energy, (forces, shapes_meta, param_res)


def _pallas_bonded_and_nonbonded_nl_fused_bwd(
    has_box,
    pme_alpha,
    cutoff,
    coulomb_constant,
    lj_switch_width,
    softcore_alpha,
    softcore_exponent,
    include_pme_excl_erf,
    interpret,
    excl_mode,
    scale_14_vdw,
    scale_14_elec,
    dist_eps,
    b0_is_ji,
    angle_regularization_mode,
    cmap_first_is_psi,
    compute_param_grads,
    res,
    g,
):
    del interpret, excl_mode, angle_regularization_mode, compute_param_grads
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
        charges,
        sigmas,
        epsilons,
        atom_mask,
        neighbor_idx,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
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
        lam_arr,
    ) = param_res

    if positions.ndim == 2:
        gq, gs, ge, gev, gee, glam_nb = _nonbonded_nl_param_grads(
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
            scale_14_vdw,
            scale_14_elec,
        )
        gb, ga, gd, gh, ges, gee14, geq, gc, glam_b = _bonded_all_fused_param_grads(
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
            lam_arr.reshape(()),
            has_box,
            dist_eps,
            b0_is_ji,
            coulomb_constant,
            softcore_alpha,
            softcore_exponent,
            cmap_first_is_psi,
        )
        grad_q = (g * gq).astype(charges.dtype)
        grad_sig = (g * gs).astype(sigmas.dtype)
        grad_eps = (g * ge).astype(epsilons.dtype)
        grad_ev = (g * gev).astype(excl_scales_vdw.dtype)
        grad_ee = (g * gee).astype(excl_scales_elec.dtype)
        grad_bp = (g * gb).astype(bond_params.dtype)
        grad_ap = (g * ga).astype(angle_params.dtype)
        grad_dp = (g * gd).astype(dih_params.dtype)
        grad_hp = (g * gh).astype(harm_imp_params.dtype)
        grad_es14 = (g * ges).astype(exc14_sigmas.dtype)
        grad_ee14 = (g * gee14).astype(exc14_epsilons.dtype)
        grad_eq14 = (g * geq).astype(exc14_chargeprods.dtype)
        grad_cc = (g * gc).astype(cmap_coeffs.dtype)
        grad_lm = (g * (glam_nb + glam_b.reshape(lam_arr.shape))).astype(lam_arr.dtype)
    else:
        B = positions.shape[0]
        in_b = tuple(bool(a.ndim > bn) for a, bn in zip(param_res, _MEGAKERNEL_BASE_NDIMS))

        def _step(b):
            s = [a[b] if ib else a for a, ib in zip(param_res, in_b)]
            gq_b, gs_b, ge_b, gev_b, gee_b, glam_nb_b = _nonbonded_nl_param_grads(
                s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[30], s[31],
                has_box, pme_alpha, cutoff, coulomb_constant, lj_switch_width,
                softcore_alpha, softcore_exponent, include_pme_excl_erf,
                scale_14_vdw, scale_14_elec,
            )
            gb_b, ga_b, gd_b, gh_b, ges_b, gee14_b, geq_b, gc_b, glam_b_b = _bonded_all_fused_param_grads(
                s[0], s[9], s[10], s[11], s[12], s[13], s[14], s[15], s[16], s[17],
                s[18], s[19], s[20], s[21], s[22], s[23], s[24], s[25], s[26], s[27],
                s[28], s[29], s[30], s[31].reshape(()),
                has_box, dist_eps, b0_is_ji, coulomb_constant,
                softcore_alpha, softcore_exponent, cmap_first_is_psi,
            )
            g_b = g[b] if g.ndim > 0 else g
            return (
                (g_b * gq_b).astype(charges.dtype),
                (g_b * gs_b).astype(sigmas.dtype),
                (g_b * ge_b).astype(epsilons.dtype),
                (g_b * gev_b).astype(excl_scales_vdw.dtype),
                (g_b * gee_b).astype(excl_scales_elec.dtype),
                (g_b * gb_b).astype(bond_params.dtype),
                (g_b * ga_b).astype(angle_params.dtype),
                (g_b * gd_b).astype(dih_params.dtype),
                (g_b * gh_b).astype(harm_imp_params.dtype),
                (g_b * ges_b).astype(exc14_sigmas.dtype),
                (g_b * gee14_b).astype(exc14_epsilons.dtype),
                (g_b * geq_b).astype(exc14_chargeprods.dtype),
                (g_b * gc_b).astype(cmap_coeffs.dtype),
                (g_b * (glam_nb_b + glam_b_b.reshape(s[31].shape))).astype(lam_arr.dtype),
            )

        (
            gq_all, gs_all, ge_all, gev_all, gee_all,
            gb_all, ga_all, gd_all, gh_all,
            ges_all, gee14_all, geq_all, gc_all, glam_all,
        ) = jax.lax.map(_step, jnp.arange(B))
        grad_q = gq_all if in_b[1] else jnp.sum(gq_all, axis=0)
        grad_sig = gs_all if in_b[2] else jnp.sum(gs_all, axis=0)
        grad_eps = ge_all if in_b[3] else jnp.sum(ge_all, axis=0)
        grad_ev = gev_all if in_b[7] else jnp.sum(gev_all, axis=0)
        grad_ee = gee_all if in_b[8] else jnp.sum(gee_all, axis=0)
        grad_bp = gb_all if in_b[10] else jnp.sum(gb_all, axis=0)
        grad_ap = ga_all if in_b[13] else jnp.sum(ga_all, axis=0)
        grad_dp = gd_all if in_b[16] else jnp.sum(gd_all, axis=0)
        grad_hp = gh_all if in_b[19] else jnp.sum(gh_all, axis=0)
        grad_es14 = ges_all if in_b[22] else jnp.sum(ges_all, axis=0)
        grad_ee14 = gee14_all if in_b[23] else jnp.sum(gee14_all, axis=0)
        grad_eq14 = geq_all if in_b[24] else jnp.sum(geq_all, axis=0)
        grad_cc = gc_all if in_b[28] else jnp.sum(gc_all, axis=0)
        grad_lm = glam_all if in_b[31] else jnp.sum(glam_all, axis=0)

    z = lambda idx: jnp.zeros(shapes_meta[idx].shape, dtype=shapes_meta[idx].dtype)
    return (
        grad_pos,
        grad_q,
        grad_sig,
        grad_eps,
        z(4),
        z(5),
        z(6),
        grad_ev,
        grad_ee,
        z(9),
        grad_bp,
        z(11),
        z(12),
        grad_ap,
        z(14),
        z(15),
        grad_dp,
        z(17),
        z(18),
        grad_hp,
        z(20),
        z(21),
        grad_es14,
        grad_ee14,
        grad_eq14,
        z(25),
        z(26),
        z(27),
        grad_cc,
        z(29),
        z(30),
        grad_lm,
    )


_pallas_bonded_and_nonbonded_nl_fused_diff.defvjp(
    _pallas_bonded_and_nonbonded_nl_fused_fwd,
    _pallas_bonded_and_nonbonded_nl_fused_bwd,
    symbolic_zeros=True,
)


import prolix.physics.pallas_bonded as _pallas_bonded_mod

_ORIG_PALLAS_NONBONDED_NL_BATCH = _pallas_nonbonded_nl_batch
_ORIG_PALLAS_BONDED_ALL_FUSED_BATCH = _pallas_bonded_mod._pallas_bonded_all_fused_batch


def pallas_bonded_and_nonbonded_nl_fused(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array | None,
    neighbor_idx: jax.Array,
    excl_indices: jax.Array | None = None,
    excl_scales_vdw: jax.Array | None = None,
    excl_scales_elec: jax.Array | None = None,
    *,
    bonds: jax.Array | None = None,
    bond_params: jax.Array | None = None,
    bond_mask: jax.Array | None = None,
    urey_bradley_bonds: jax.Array | None = None,
    urey_bradley_params: jax.Array | None = None,
    urey_bradley_mask: jax.Array | None = None,
    angles: jax.Array | None = None,
    angle_params: jax.Array | None = None,
    angle_mask: jax.Array | None = None,
    dihedrals: jax.Array | None = None,
    dihedral_params: jax.Array | None = None,
    dihedral_mask: jax.Array | None = None,
    impropers: jax.Array | None = None,
    improper_params: jax.Array | None = None,
    improper_mask: jax.Array | None = None,
    improper_is_harmonic: bool | None = None,
    exception_pairs: jax.Array | None = None,
    exception_sigmas: jax.Array | None = None,
    exception_epsilons: jax.Array | None = None,
    exception_chargeprods: jax.Array | None = None,
    exception_mask: jax.Array | None = None,
    cmap_torsions: jax.Array | None = None,
    cmap_indices: jax.Array | None = None,
    cmap_coeffs: jax.Array | None = None,
    cmap_mask: jax.Array | None = None,
    cmap_first_is_psi: bool = True,
    box: jax.Array | None = None,
    pme_alpha: float = 0.0,
    cutoff: float = 10.0,
    coulomb_constant: float = constants.COULOMB_CONSTANT,
    lj_switch_width: float = 0.0,
    soft_core_lambda: float | jax.Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    include_pme_excl_erf: bool = True,
    excl_mode: str = "table",
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    dist_eps: float = 1e-10,
    b0_is_ji: bool = True,
    angle_regularization_mode: str = "legacy_denom_eps",
    compute_param_grads: bool | None = None,
    return_forces: bool = False,
    interpret: bool | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Single-launch fused direct-space nonbonded NL + all-bonded Pallas megakernel (MK-36)."""
    _batch_spied = (
        not return_forces
        and (
            _pallas_nonbonded_nl_batch is not _ORIG_PALLAS_NONBONDED_NL_BATCH
            or _pallas_bonded_mod._pallas_bonded_all_fused_batch
            is not _ORIG_PALLAS_BONDED_ALL_FUSED_BATCH
        )
    )
    if in_export_context() or _batch_spied:
        e_b = pallas_bonded_all_fused(
            positions,
            bonds=bonds,
            bond_params=bond_params,
            bond_mask=bond_mask,
            urey_bradley_bonds=urey_bradley_bonds,
            urey_bradley_params=urey_bradley_params,
            urey_bradley_mask=urey_bradley_mask,
            angles=angles,
            angle_params=angle_params,
            angle_mask=angle_mask,
            dihedrals=dihedrals,
            dihedral_params=dihedral_params,
            dihedral_mask=dihedral_mask,
            impropers=impropers,
            improper_params=improper_params,
            improper_mask=improper_mask,
            improper_is_harmonic=improper_is_harmonic,
            exception_pairs=exception_pairs,
            exception_sigmas=exception_sigmas,
            exception_epsilons=exception_epsilons,
            exception_chargeprods=exception_chargeprods,
            exception_mask=exception_mask,
            cmap_torsions=cmap_torsions,
            cmap_indices=cmap_indices,
            cmap_coeffs=cmap_coeffs,
            cmap_mask=cmap_mask,
            cmap_first_is_psi=cmap_first_is_psi,
            box=box,
            dist_eps=dist_eps,
            b0_is_ji=b0_is_ji,
            angle_regularization_mode=angle_regularization_mode,
            coulomb_constant=coulomb_constant,
            soft_core_lambda=soft_core_lambda,
            softcore_alpha=softcore_alpha,
            softcore_exponent=softcore_exponent,
            compute_param_grads=compute_param_grads,
        )
        e_nb = pallas_nonbonded_nl(
            positions,
            charges,
            sigmas,
            epsilons,
            atom_mask,
            neighbor_idx,
            excl_indices,
            excl_scales_vdw,
            excl_scales_elec,
            box=box,
            pme_alpha=pme_alpha,
            cutoff=cutoff,
            coulomb_constant=coulomb_constant,
            lj_switch_width=lj_switch_width,
            soft_core_lambda=soft_core_lambda,
            softcore_alpha=softcore_alpha,
            softcore_exponent=softcore_exponent,
            include_pme_excl_erf=include_pme_excl_erf,
            excl_mode=excl_mode,
            scale_14_vdw=scale_14_vdw,
            scale_14_elec=scale_14_elec,
            compute_param_grads=compute_param_grads,
        )
        return e_b + e_nb

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

    q_arr = jnp.asarray(charges, dtype=dtype)
    sig_arr = jnp.asarray(sigmas, dtype=dtype)
    eps_arr = jnp.asarray(epsilons, dtype=dtype)
    nbr_arr = jnp.asarray(neighbor_idx, dtype=jnp.int32)

    # 1. Consolidate 2-body harmonic bonds + Urey-Bradley
    b_idx_parts, b_prm_parts, b_msk_parts = [], [], []
    if bonds is not None and bond_params is not None and bonds.shape[-2] > 0:
        bi = jnp.asarray(bonds[..., :, :2], dtype=jnp.int32)
        bp = jnp.asarray(bond_params, dtype=dtype)
        bm = jnp.ones(bi.shape[:-1], dtype=dtype) if bond_mask is None else jnp.asarray(bond_mask, dtype=dtype)
        b_idx_parts.append(bi)
        b_prm_parts.append(bp)
        b_msk_parts.append(bm)
    if urey_bradley_bonds is not None and urey_bradley_params is not None and urey_bradley_bonds.shape[-2] > 0:
        ui = jnp.asarray(urey_bradley_bonds[..., :, :2], dtype=jnp.int32)
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
    if angles is not None and angle_params is not None and angles.shape[-2] > 0:
        a_idx_arr = jnp.asarray(angles[..., :, :3], dtype=jnp.int32)
        a_prm_arr = jnp.asarray(angle_params, dtype=dtype)
        a_msk_arr = jnp.ones(a_idx_arr.shape[:-1], dtype=dtype) if angle_mask is None else jnp.asarray(angle_mask, dtype=dtype)
    else:
        a_idx_arr = jnp.zeros((0, 3), dtype=jnp.int32)
        a_prm_arr = jnp.zeros((0, 2), dtype=dtype)
        a_msk_arr = jnp.zeros((0,), dtype=dtype)

    # 3 & 4. Consolidate periodic dihedrals + periodic impropers vs harmonic impropers
    d_idx_parts, d_prm_parts, d_msk_parts = [], [], []
    h_idx_parts, h_prm_parts, h_msk_parts = [], [], []
    if dihedrals is not None and dihedral_params is not None and dihedrals.shape[-2] > 0:
        di = jnp.asarray(dihedrals[..., :, :4], dtype=jnp.int32)
        dp = jnp.asarray(dihedral_params, dtype=dtype)
        dm = jnp.ones(di.shape[:-1], dtype=dtype) if dihedral_mask is None else jnp.asarray(dihedral_mask, dtype=dtype)
        di, dp, dm = _flatten_multi_term_torsion(positions, di, dp, dm)
        d_idx_parts.append(di)
        d_prm_parts.append(dp)
        d_msk_parts.append(dm)
    if impropers is not None and improper_params is not None and impropers.shape[-2] > 0:
        ii = jnp.asarray(impropers[..., :, :4], dtype=jnp.int32)
        ip = jnp.asarray(improper_params, dtype=dtype)
        im = jnp.ones(ii.shape[:-1], dtype=dtype) if improper_mask is None else jnp.asarray(improper_mask, dtype=dtype)
        ii, ip, im = _flatten_multi_term_torsion(positions, ii, ip, im)
        is_harm_imp = bool(improper_is_harmonic) if improper_is_harmonic is not None else (ip.shape[-1] < 3)
        if is_harm_imp:
            h_idx_parts.append(ii)
            h_prm_parts.append(ip[..., :2])
            h_msk_parts.append(im)
        else:
            d_idx_parts.append(ii)
            d_prm_parts.append(ip[..., :3])
            d_msk_parts.append(im)

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
            if cmap_indices is None
            else jnp.asarray(cmap_indices, dtype=jnp.int32)
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

    box_arr, has_box = _normalize_box(box, positions)
    lam_raw = jnp.asarray(soft_core_lambda, dtype=dtype)
    lam_arr = lam_raw.reshape((1,)) if lam_raw.ndim == 0 else lam_raw.reshape((-1, 1))

    if interpret is None:
        interpret = not _triton_gpu_available()

    arr_args = (
        positions,
        q_arr,
        sig_arr,
        eps_arr,
        msk_arr,
        nbr_arr,
        ei_arr,
        ev_arr,
        ee_arr,
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
    static_args = (
        bool(has_box),
        float(pme_alpha),
        float(cutoff),
        float(coulomb_constant),
        float(lj_switch_width),
        float(softcore_alpha),
        int(softcore_exponent),
        bool(include_pme_excl_erf),
        bool(interpret),
        str(excl_mode),
        float(scale_14_vdw),
        float(scale_14_elec),
        float(dist_eps),
        bool(b0_is_ji),
        str(angle_regularization_mode),
        bool(cmap_first_is_psi),
    )

    if return_forces:
        return _pallas_bonded_and_nonbonded_nl_fused_vmappable(*arr_args, *static_args)

    return _pallas_bonded_and_nonbonded_nl_fused_diff(
        *arr_args,
        *static_args,
        compute_param_grads,
    )


__all__ = [
    "PallasKernelConfig",
    "angles_harmonic_reference",
    "bonded_all_fused_reference",
    "bonds_2body_reference",
    "born_radii_reference",
    "cmap_torsions_reference",
    "compute_morton_codes_3d",
    "encode_nl_exclusions",
    "exceptions_14_reference",
    "flash_explicit_nonbonded_reference",
    "gb_ace_nl_reference",
    "gb_coulomb_energy_dense",
    "gb_energy_split_cutoff",
    "make_angles_harmonic_op",
    "make_bonded_all_fused_op",
    "make_bonds_2body_op",
    "make_born_radii_op",
    "make_cmap_torsions_op",
    "make_exceptions_14_op",
    "make_flash_explicit_nonbonded_op",
    "make_gb_ace_nl_op",
    "make_nonbonded_nl_op",
    "make_pme_exclusion_pairs_op",
    "make_sasa_nl_op",
    "make_torsions_4body_op",
    "nonbonded_nl_reference",
    "pallas_angles_harmonic",
    "pallas_bonded_all_fused",
    "pallas_bonded_and_nonbonded_nl_fused",
    "pallas_bonds_2body",
    "pallas_born_radii",
    "pallas_cmap_torsions",
    "pallas_exceptions_14",
    "pallas_flash_explicit_nonbonded",
    "pallas_gb_ace_nl",
    "pallas_nonbonded_nl",
    "pallas_pme_exclusion_pairs",
    "pallas_prune_and_encode_cell_list",
    "pallas_sasa_nl",
    "pallas_torsions_4body",
    "permute_and_sort_exclusions",
    "permute_neighbor_list_morton",
    "pme_exclusion_pairs_reference",
    "resolve_pallas_mode",
    "sasa_nl_reference",
    "torsions_4body_reference",
]
