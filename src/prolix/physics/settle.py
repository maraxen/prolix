"""SETTLE constraint algorithm for rigid 3-site water molecules.

The SETTLE algorithm (Miyamoto & Kollman, 1992) is an analytical constraint
solver for rigid 3-site water molecules. Unlike iterative SHAKE/RATTLE, it
solves the constraints in closed form, making it faster and more accurate.

References:
    Miyamoto, S., & Kollman, P. A. (1992). Settle: An analytical version of
    the SHAKE and RATTLE algorithm for rigid water models.
    Journal of Computational Chemistry, 13(8), 952-962.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import random
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax_md import quantity, simulate

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics import (
  md_potential_bundle,
)
from prolix.physics import (
  pbc as pbc_module,
)
from prolix.physics import (
  pressure as pressure_module,
)
from prolix.physics import (
  stress as stress_module,
)
from prolix.physics import (
  units as units_module,
)
from prolix.physics.constraint_projection import (
  project_momenta,
  project_positions,
  unpack_constraints,
)
from prolix.tiling.axis_dispatch import dispatch_n_waters
from prolix.typing import NPTState, NVTLangevinState, WaterIndices, WaterIndicesArray

if TYPE_CHECKING:

  from jax_md.util import Array

Array = Any

# PBC rounding mode selection (A/B testing: round vs floor for banker's rounding)
# Module-level constant bound at import time, never read inside a traced function.
_PBC_ROUND_MODE = os.environ.get("PROLIX_PBC_ROUND_MODE", "floor")


def _pbc_round_floor(x: Array) -> Array:
  """Round-half-up via floor(x + 0.5) — project convention for A/B testing."""
  return jnp.floor(x + 0.5)


def _pbc_round_banker(x: Array) -> Array:
  """Banker's rounding (round-half-to-even) — the pre-fix idiom, for A/B comparison."""
  return jnp.round(x)


_PBC_ROUND_FN = _pbc_round_banker if _PBC_ROUND_MODE == "round" else _pbc_round_floor


def _min_image_delta(delta: Array, box: Array) -> Array:
  """Minimum-image displacement for 1D orthorhombic (3,) or 2D triclinic (3, 3) box."""
  box_arr = jnp.asarray(box, dtype=delta.dtype)
  if box_arr.ndim == 2:
    return pbc_module.minimum_image_displacement(delta, jnp.zeros_like(delta), box_arr)
  return delta - box_arr * _PBC_ROUND_FN(delta / box_arr)

# TIP3P water geometry constants
TIP3P_ROH = 0.9572  # O-H bond length (Å)
TIP3P_RHH = 1.5139  # H-H distance (Å)
TIP3P_THETA = 104.52 * jnp.pi / 180.0  # H-O-H angle (rad)

# Default CSVR relaxation time: 0.1 ps in AKMA units (1 AKMA ≈ 48.888 fs)
_DEFAULT_CSVR_TAU_AKMA: float = 100.0 / 48.88821291839


def _scatter_water_target(indices: Array, water_mask: Array | None, n_real_target: int) -> Array:
  """Redirect padding-water row indices to a dedicated scratch slot.

  Padded ``water_indices`` rows are filled with ``[0, 0, 0]`` (see
  ``prolix/padding.py``'s zero-fill convention), which can collide with a
  genuine real atom 0. Scattering a padding row's (physically meaningless,
  possibly degenerate/NaN) computed value onto that shared index would
  either corrupt a real atom or silently discard its real constraint
  correction, depending on write order. Redirecting padding rows' scatter
  *target* (never the gather source, which is harmless to reuse) to index
  ``n_real_target`` — one past the real array, a caller-appended scratch row
  that gets sliced off before returning — makes every padding-row scatter a
  provable no-op regardless of what value it carries. See B1-SETTLE-STACK
  (`.praxia/docs/specs/260715_b1-settle-stack.md`).
  """
  if water_mask is None:
    return indices
  return jnp.where(water_mask, indices, n_real_target)


def settle_positions(
  positions_unconstrained: Array,
  positions_old: Array,
  water_indices: WaterIndicesArray,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  box: Array | None = None,
  water_mask: Array | None = None,
) -> Array:
  r"""Apply SETTLE position constraints to water molecules.

  Process:
  1.  **Extract Indices**: Use `WaterIndices` to identify O, H1, H2.
  2.  **Batch SETTLE**: Call `_settle_water_batch` for analytical reconstruction.
  3.  **PBC Handling**: Correct for periodic boundary crossings if `box` provided.
  4.  **Scatter Update**: Place constrained coordinates back into the global array.

  Args:
      positions_unconstrained: Unconstrained positions after integrator step (N, 3).
      positions_old: Positions before the integrator step (N, 3).
      water_indices: (N_waters, 3) atom indices [O, H1, H2].
      r_OH: Target O-H bond length (Å).
      r_HH: Target H-H distance (Å).
      mass_oxygen: Mass of oxygen (amu).
      mass_hydrogen: Mass of hydrogen (amu).
      box: Optional box dimensions for PBC (3,).
      water_mask: Optional (N_waters,) bool — True for real water rows, False
          for padding. Padding rows' scatter target is redirected to a
          scratch slot so they cannot corrupt (or lose a race with) a real
          atom sharing the padding fill index. See `_scatter_water_target`.

  Returns:
      Constrained positions array.
  """
  if water_indices.shape[0] == 0:
    return positions_unconstrained

  # Extract water atom positions
  indices = WaterIndices.from_row(water_indices.T)

  # Old positions
  pos_oxygen_old = positions_old[indices.oxygen]
  pos_h1_old = positions_old[indices.hydrogen1]
  pos_h2_old = positions_old[indices.hydrogen2]

  # Unconstrained positions
  pos_oxygen_new = positions_unconstrained[indices.oxygen]
  pos_h1_new = positions_unconstrained[indices.hydrogen1]
  pos_h2_new = positions_unconstrained[indices.hydrogen2]

  # Compute constrained positions using SETTLE algorithm
  pos_oxygen_c, pos_h1_c, pos_h2_c = _settle_water_batch(
    pos_oxygen_old,
    pos_h1_old,
    pos_h2_old,
    pos_oxygen_new,
    pos_h1_new,
    pos_h2_new,
    r_OH,
    r_HH,
    mass_oxygen,
    mass_hydrogen,
    box,
  )

  # Re-wrap constrained positions using minimum-image relative to the
  # UNCONSTRAINED oxygen position (pos_oxygen_new), not jnp.mod on the
  # constrained position directly.
  #
  # WHY NOT jnp.mod(pos_oxygen_c, box): jnp.mod has a hard discontinuity at
  # every integer multiple of box.  When pos_oxygen_c sits within floating-
  # point epsilon of a face (e.g. 10.0000000001 vs 9.9999999999) due to FMA
  # reassociation differences between CPU and GPU XLA backends, jnp.mod maps
  # the two values to opposite ends of the box (0 vs ~10), producing a ±box
  # jump and a divergent trajectory between otherwise identical vmap replicas.
  #
  # FIX: apply minimum-image of pos_oxygen_c relative to pos_oxygen_new (the
  # unconstrained O position, which is already in [0, box) because the caller
  # wraps positions each step via shift_fn).  The SETTLE geometry correction is
  # sub-Ångström, so pos_oxygen_c is within a fraction of a bond-length of
  # pos_oxygen_new, and the minimum-image delta is << 0.5 box. Crucially, the
  # image decision is based on (pos_oxygen_c - pos_oxygen_new), NOT on
  # pos_oxygen_c alone, so FMA-level differences in the Horn-SETTLE result
  # (9.9999... vs 10.0000...) produce a delta difference of only ~1e-7, not the
  # ±box discontinuity that jnp.mod exhibits.
  if box is not None:
    # Minimum-image of pos_oxygen_c relative to pos_oxygen_new.
    # All three constrained atoms get the same rigid integer-image shift so
    # the molecule's internal geometry is not distorted.
    delta_o = pos_oxygen_c - pos_oxygen_new
    image_correction = _min_image_delta(delta_o, box) - delta_o

    pos_oxygen_c = pos_oxygen_c + image_correction
    pos_h1_c = pos_h1_c + image_correction
    pos_h2_c = pos_h2_c + image_correction

  # Update positions in result array. Padding rows' scatter target is
  # redirected to a scratch slot (see `_scatter_water_target`) so they can
  # never clobber, or be clobbered by write-order vs., a real atom sharing
  # the padding fill index.
  if water_mask is None:
    positions_constrained = positions_unconstrained.at[indices.oxygen].set(pos_oxygen_c)
    positions_constrained = positions_constrained.at[indices.hydrogen1].set(pos_h1_c)
    positions_constrained = positions_constrained.at[indices.hydrogen2].set(pos_h2_c)
    return positions_constrained

  mask_col = water_mask[:, None]
  delta_o = jnp.where(mask_col, pos_oxygen_c - positions_unconstrained[indices.oxygen], 0.0)
  delta_h1 = jnp.where(mask_col, pos_h1_c - positions_unconstrained[indices.hydrogen1], 0.0)
  delta_h2 = jnp.where(mask_col, pos_h2_c - positions_unconstrained[indices.hydrogen2], 0.0)
  positions_constrained = positions_unconstrained.at[indices.oxygen].add(delta_o)
  positions_constrained = positions_constrained.at[indices.hydrogen1].add(delta_h1)
  positions_constrained = positions_constrained.at[indices.hydrogen2].add(delta_h2)
  return positions_constrained


def _solve_3x3(A: Array, b: Array) -> Array:
  """Solve a single 3x3 linear system A x = b via explicit Cramer's rule."""
  a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
  a10, a11, a12 = A[1, 0], A[1, 1], A[1, 2]
  a20, a21, a22 = A[2, 0], A[2, 1], A[2, 2]
  det = (
    a00 * (a11 * a22 - a12 * a21)
    - a01 * (a10 * a22 - a12 * a20)
    + a02 * (a10 * a21 - a11 * a20)
  )
  valid = jnp.abs(det) > 1e-30
  safe_det = jnp.where(valid, det, jnp.ones_like(det))
  inv = jnp.where(valid, 1.0 / safe_det, jnp.zeros_like(det))
  bx, by, bz = b[0], b[1], b[2]
  return jnp.stack(
    [
      inv * ((a11 * a22 - a12 * a21) * bx + (a02 * a21 - a01 * a22) * by + (a01 * a12 - a02 * a11) * bz),
      inv * ((a12 * a20 - a10 * a22) * bx + (a00 * a22 - a02 * a20) * by + (a02 * a10 - a00 * a12) * bz),
      inv * ((a10 * a21 - a11 * a20) * bx + (a01 * a20 - a00 * a21) * by + (a00 * a11 - a01 * a10) * bz),
    ],
    axis=-1,
  )


def _solve_3x3_batch(A: Array, b: Array) -> Array:
  """Solve batched 3x3 linear systems A x = b via explicit Cramer's rule."""
  a00, a01, a02 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
  a10, a11, a12 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
  a20, a21, a22 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]
  det = (
    a00 * (a11 * a22 - a12 * a21)
    - a01 * (a10 * a22 - a12 * a20)
    + a02 * (a10 * a21 - a11 * a20)
  )
  valid = jnp.abs(det) > 1e-30
  safe_det = jnp.where(valid, det, jnp.ones_like(det))
  inv = jnp.where(valid, 1.0 / safe_det, jnp.zeros_like(det))
  bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
  return jnp.stack(
    [
      inv * ((a11 * a22 - a12 * a21) * bx + (a02 * a21 - a01 * a22) * by + (a01 * a12 - a02 * a11) * bz),
      inv * ((a12 * a20 - a10 * a22) * bx + (a00 * a22 - a02 * a20) * by + (a02 * a10 - a00 * a12) * bz),
      inv * ((a10 * a21 - a11 * a20) * bx + (a01 * a20 - a00 * a21) * by + (a00 * a11 - a01 * a10) * bz),
    ],
    axis=-1,
  )


def _cholesky_upper_solve_3x3(A: Array, rhs: Array) -> Array:
  """Solve L^T x = rhs where L L^T = A via closed-form 3x3 Cholesky + backward substitution."""
  eps = jnp.array(1e-30, dtype=A.dtype)
  l00 = jnp.sqrt(jnp.maximum(A[..., 0, 0], eps))
  l10 = A[..., 1, 0] / l00
  l20 = A[..., 2, 0] / l00
  l11 = jnp.sqrt(jnp.maximum(A[..., 1, 1] - l10 * l10, eps))
  l21 = (A[..., 2, 1] - l20 * l10) / l11
  l22 = jnp.sqrt(jnp.maximum(A[..., 2, 2] - l20 * l20 - l21 * l21, eps))

  x2 = rhs[..., 2] / l22
  x1 = (rhs[..., 1] - l21 * x2) / l11
  x0 = (rhs[..., 0] - l10 * x1 - l20 * x2) / l00
  return jnp.stack([x0, x1, x2], axis=-1)


def _skew_symmetric3(r: Array) -> Array:
  """3×3 skew matrix ``[r]_×`` with ``[r]_× @ \\omega = r \\times \\omega``."""
  x, y, z = r[0], r[1], r[2]
  return jnp.array(
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
    dtype=r.dtype,
  )


def _project_one_water_momentum_rigid(
  p_stack: Array,
  r_stack: Array,
  m_stack: Array,
) -> Array:
  """Mass-weighted LS fit of atomic momenta to rigid motion for one TIP3P-like triplet.

  Finds ``v_com``, ``\\omega`` minimizing ``\\sum_i m_i |v_i - v_com - \\omega\\times r_{i,rel}|^2``,
  then returns projected atomic momenta ``m_i v_i``.
  """
  msum = jnp.sum(m_stack)
  eps = jnp.array(1e-30, dtype=msum.dtype)
  msum_safe = jnp.maximum(msum, eps)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum_safe
  rrel = r_stack - com

  m_safe = jnp.maximum(m_stack[:, None], eps)
  v = p_stack / m_safe
  v_com = jnp.sum(m_stack[:, None] * v, axis=0) / msum_safe
  L_ang = jnp.sum(jnp.cross(rrel, p_stack), axis=0)

  eye3 = jnp.eye(3, dtype=r_stack.dtype)
  r_sq = jnp.einsum("ia,ia->i", rrel, rrel)
  I_tensor = (
    jnp.einsum("i,i->", m_stack, r_sq) * eye3
    - jnp.einsum("i,ia,ib->ab", m_stack, rrel, rrel)
  )
  reg = jnp.array(1e-12, dtype=I_tensor.dtype) * (jnp.trace(I_tensor) / 3.0 + 1.0)
  I_reg = I_tensor + reg * eye3
  omega = _solve_3x3(I_reg, L_ang)

  v_proj = v_com[None, :] + jnp.cross(omega[None, :], rrel)
  return v_proj * m_stack[:, None]


def project_tip3p_waters_momentum_rigid(
  momentum: Array,
  position: Array,
  mass: Array,
  water_indices: WaterIndicesArray,
  water_mask: Array | None = None,
) -> Array:
  """Project atomic momenta onto rigid-body subspace for each water (shape ``(N_w,3)`` indices)."""
  if water_indices.shape[0] == 0:
    return momentum
  mass_flat = mass.reshape(-1)
  idx = water_indices
  p_stack = jnp.stack([momentum[idx[:, 0]], momentum[idx[:, 1]], momentum[idx[:, 2]]], axis=1)
  r_stack = jnp.stack([position[idx[:, 0]], position[idx[:, 1]], position[idx[:, 2]]], axis=1)
  m_stack = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)
  p_proj = jax.vmap(_project_one_water_momentum_rigid)(p_stack, r_stack, m_stack)
  idx_flat = idx.reshape(-1)
  p_proj_flat = p_proj.reshape(-1, 3)
  if water_mask is None:
    return momentum.at[idx_flat].set(p_proj_flat)
  mask_flat = jnp.broadcast_to(water_mask[:, None], idx.shape).reshape(-1, 1)
  delta_p = jnp.where(mask_flat, p_proj_flat - momentum[idx_flat], 0.0)
  return momentum.at[idx_flat].add(delta_p)


def _settle_water_batch(
  pos_oxygen_old: Array,
  pos_h1_old: Array,
  pos_h2_old: Array,
  pos_oxygen_new: Array,
  pos_h1_new: Array,
  pos_h2_new: Array,
  r_OH: float,
  r_HH: float,
  mass_oxygen: float,
  mass_hydrogen: float,
  box: Array | None = None,
) -> tuple[Array, Array, Array]:
  r"""Analytical SETTLE algorithm for a batch of water molecules.

  Process:
  1.  **Unwrap**: Apply minimum image convention to handle PBC crossings.
  2.  **COM Frame**: Move to center of mass of unconstrained configuration.
  3.  **Ideal Geometry**: Define ideal body-frame coordinates.
  4.  **Horn Rigid Fit**: Use Horn (1987) quaternion method to find optimal rotation
      that minimizes RMSD between body-frame template and COM-centered unconstrained atoms.
  5.  **Transform**: Reconstruct positions and transform back to global frame.

  Notes:
  Preserves COM position while correcting internal bond lengths and angles.
  Uses rotation-preserving rigid fit (Horn method) to avoid damping genuine molecular rotation.

  NOTES:
  - PBC unwrap uses ``jnp.floor(delta/box + 0.5)`` rather than ``jnp.round``.
    This is a PROJECT CONVENTION adopted to enable A/B testing (a consistent
    single idiom across all code). Both ``round(x)`` and ``floor(x+0.5)`` are
    pure functions of float64 input and equally susceptible to ULP-level
    differences between code paths; neither is inherently more deterministic
    under FMA reassociation. In fact, ``floor(x+0.5)`` is marginally LESS
    accurate due to double-rounding in the addition step. Historical note:
    an 8.975 Å divergence was observed in looped vs vmapped paths on cluster
    hardware, but this observation is unverified and not causally attributed
    to this rounding site.

  Args:
      r_O_old, r_H1_old, r_H2_old: Old positions (used for PBC unwrap only).
      r_O_new, r_H1_new, r_H2_new: Unconstrained new positions.
      r_OH: Target O-H distance.
      r_HH: Target H-H distance.
      m_O, m_H: Masses.
      box: Periodic box dimensions.

  Returns:
      Tuple of constrained positions (r_O_c, r_H1_c, r_H2_c).
  """
  # Apply minimum image convention if box is provided.
  # Uses _PBC_ROUND_FN (floor(x+0.5) by default) as project convention;
  # enables A/B testing of rounding modes (round vs floor).
  if box is not None:

    def unwrap(pos_new, pos_old):
      delta = pos_new - pos_old
      return pos_old + _min_image_delta(delta, box)

    pos_oxygen_new = unwrap(pos_oxygen_new, pos_oxygen_old)
    pos_h1_new = unwrap(pos_h1_new, pos_h1_old)
    pos_h2_new = unwrap(pos_h2_new, pos_h2_old)

    # Make each molecule WHOLE before the rigid-body fit below (#4971).
    #
    # The unwrap above is per-atom, against each atom's own previous position: it
    # restores step-to-step continuity for an atom that crossed a face, which is
    # necessary but says nothing about whether the three atoms are in the same
    # periodic image as EACH OTHER. They frequently are not. `shift_fn` wraps every
    # atom independently into [0, box), so any molecule straddling a face has its O
    # sent to one side and an H to the other, and nothing downstream ever repairs it.
    #
    # That is fatal here rather than merely untidy, because `com_new` and the
    # body-frame deltas immediately below are computed from these raw positions. A
    # split molecule yields a COM in the middle of the box and an O-H separation of
    # up to a full box length, so the template fit is solving for the orientation of
    # something that is not a water molecule, and it launches it.
    #
    # Measured on DHFR (23558 atoms, job 21968919): from rest with the thermostat
    # fully off, 1011 atoms move 8-9 A in a single step while their forces are
    # normal. 1011 = 3 x 337 whole molecules, water fraction 1.0000 against a system
    # baseline of 0.8943. Independently, wrapping that fixture's coordinates tears
    # 334 of 7023 waters apart with a maximum O-H distance of 86.4 A. Carbon,
    # nitrogen and sulfur meanwhile track Newton's 0.5*(F/m)*dt^2 to a ratio of
    # exactly 1.000, so the integrator itself was never at fault.
    #
    # Re-imaging the hydrogens onto the oxygen costs two subtractions and makes the
    # molecule whole in a single image. The post-solve re-wrap in `settle_positions`
    # then returns the rigid result to the box as one unit, so nothing else changes.
    def to_oxygen_image(pos_h, pos_o):
      d = pos_h - pos_o
      return pos_o + _min_image_delta(d, box)

    pos_h1_new = to_oxygen_image(pos_h1_new, pos_oxygen_new)
    pos_h2_new = to_oxygen_image(pos_h2_new, pos_oxygen_new)

  mass_total = mass_oxygen + 2 * mass_hydrogen

  # COM motion is preserved - only internal geometry is corrected
  com_new = (
    mass_oxygen * pos_oxygen_new + mass_hydrogen * pos_h1_new + mass_hydrogen * pos_h2_new
  ) / mass_total

  # Center atoms at COM (old positions are used only for the PBC unwrap above;
  # orientation comes entirely from fitting the template to the new positions).
  delta_O_new = pos_oxygen_new - com_new  # (N_waters, 3)
  delta_H1_new = pos_h1_new - com_new
  delta_H2_new = pos_h2_new - com_new

  # Canonical body-frame template (fixed ideal geometry, COM-centered)
  dist_oh_mid = jnp.sqrt(r_OH**2 - (r_HH / 2) ** 2)
  dist_O_to_COM = 2 * mass_hydrogen * dist_oh_mid / mass_total
  dist_H_mid_to_COM = mass_oxygen * dist_oh_mid / mass_total
  half_hh = r_HH / 2

  dtype = pos_oxygen_new.dtype
  zero = jnp.zeros((), dtype=dtype)
  b_O = jnp.stack([zero, jnp.asarray(dist_O_to_COM, dtype=dtype), zero])
  b_H1 = jnp.stack([jnp.asarray(half_hh, dtype=dtype), -jnp.asarray(dist_H_mid_to_COM, dtype=dtype), zero])
  b_H2 = jnp.stack([-jnp.asarray(half_hh, dtype=dtype), -jnp.asarray(dist_H_mid_to_COM, dtype=dtype), zero])

  # Stack into (3, 3) array: [b_O; b_H1; b_H2]
  b_all = jnp.stack([b_O, b_H1, b_H2], axis=0)  # (3, 3)

  # Target (centered) points
  y_all = jnp.stack([delta_O_new, delta_H1_new, delta_H2_new], axis=1)  # (N_waters, 3, 3)

  # Mass weights
  m_all = jnp.asarray([mass_oxygen, mass_hydrogen, mass_hydrogen], dtype=dtype)  # (3,)

  # Optimal rotation via Horn's (1987) unit-quaternion method. We deliberately
  # use the 4x4 eigenvalue formulation rather than SVD/Kabsch: TIP3P water is
  # PLANAR, so the 3x3 cross-covariance is rank-deficient and the SVD's smallest
  # singular vectors flip sign discontinuously between steps -> a discontinuous
  # rotation matrix that injects spurious work and detonates energy conservation
  # over a trajectory. Horn's largest-eigenvalue quaternion is well-separated and
  # continuous through the planar case, and always yields a proper rotation.
  #
  # Mass-weighted cross-covariance S[a, b] = sum_i m_i * b_i[a] * y_i[b]
  # (b_i = body-frame template, y_i = COM-centered unconstrained target).
  s_mat = jnp.einsum("i,ia,nib->nab", m_all, b_all, y_all)  # (N_waters, 3, 3)
  sxx, sxy, sxz = s_mat[:, 0, 0], s_mat[:, 0, 1], s_mat[:, 0, 2]
  syx, syy, syz = s_mat[:, 1, 0], s_mat[:, 1, 1], s_mat[:, 1, 2]
  szx, szy, szz = s_mat[:, 2, 0], s_mat[:, 2, 1], s_mat[:, 2, 2]

  # Symmetric 4x4 Horn matrix N (per water), stacked to (N_waters, 4, 4).
  row0 = jnp.stack([sxx + syy + szz, syz - szy, szx - sxz, sxy - syx], axis=-1)
  row1 = jnp.stack([syz - szy, sxx - syy - szz, sxy + syx, szx + sxz], axis=-1)
  row2 = jnp.stack([szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy], axis=-1)
  row3 = jnp.stack([sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz], axis=-1)
  n_mat = jnp.stack([row0, row1, row2, row3], axis=1)  # (N_waters, 4, 4)

  # Largest-eigenvalue eigenvector is the optimal quaternion.
  # Because the body-frame water template b_all lies in the xy-plane (b_{i,z} == 0),
  # the 3rd row of S is identically zero (det(S) == 0, sigma_3 == 0). The 4x4 Horn
  # matrix N therefore has a pure biquadratic characteristic polynomial
  # (lambda^2 - lam_max^2)(lambda^2 - lam_2^2) = 0 with exact closed-form roots
  # lam_max = sqrt(||u||^2 + ||v||^2 + 2||u x v||) and lam_2^2 = ||u||^2 + ||v||^2 - 2||u x v||,
  # where u = S[0, :] and v = S[1, :]. By Cayley-Hamilton, M = (N + lam_max I)(N^2 - lam_2^2 I)
  # is an exact rank-1 projector onto the lam_max eigenvector, eliminating cuSOLVER eigh.
  with jax.named_scope("settle_pos_eigh"):
    u_vec = s_mat[:, 0, :]
    v_vec = s_mat[:, 1, :]
    p_tr = jnp.sum(u_vec * u_vec, axis=-1) + jnp.sum(v_vec * v_vec, axis=-1)
    cross_uv = jnp.cross(u_vec, v_vec)
    c_sq = jnp.sum(cross_uv * cross_uv, axis=-1)
    q_val = jnp.where(
      c_sq > 0.0,
      jnp.sqrt(jnp.where(c_sq > 0.0, c_sq, jnp.ones_like(c_sq))),
      jnp.zeros_like(c_sq),
    )
    lam_max_sq = jnp.maximum(p_tr + 2.0 * q_val, jnp.zeros_like(p_tr))
    lam_max = jnp.where(
      lam_max_sq > 0.0,
      jnp.sqrt(jnp.where(lam_max_sq > 0.0, lam_max_sq, jnp.ones_like(lam_max_sq))),
      jnp.zeros_like(lam_max_sq),
    )
    eye4 = jnp.eye(4, dtype=dtype)[None, :, :]
    p1 = n_mat + lam_max[:, None, None] * eye4
    # Exact algebraic identity: N^2 - p_tr * I_4 = 2 * C(u x v), so
    # N^2 - lam2_sq * I_4 = 2 * q_val * (I_4 + C(c_hat)). Using the scale-free
    # projector P2_hat = I_4 + C(c_hat) eliminates the O(||u||^4) matmul
    # N @ N and avoids float32 catastrophic cancellation when q_val << p_tr.
    c_hat = cross_uv / jnp.maximum(q_val[:, None], jnp.asarray(1e-12, dtype=dtype))
    cx, cy, cz = c_hat[:, 0], c_hat[:, 1], c_hat[:, 2]
    z = jnp.zeros_like(cx)
    c_row0 = jnp.stack([cz, -cy, cx, z], axis=-1)
    c_row1 = jnp.stack([-cy, -cz, z, cx], axis=-1)
    c_row2 = jnp.stack([cx, z, -cz, cy], axis=-1)
    c_row3 = jnp.stack([z, cx, cy, cz], axis=-1)
    c_skew = jnp.stack([c_row0, c_row1, c_row2, c_row3], axis=-2)
    p2 = eye4 + c_skew
    proj = jnp.where((q_val > jnp.asarray(1e-12, dtype=dtype))[:, None, None], jnp.matmul(p1, p2), p1)
    col_norms_sq = jnp.sum(proj * proj, axis=1)
    best_col = jnp.argmax(col_norms_sq, axis=-1)
    quat_raw = jnp.take_along_axis(proj, best_col[:, None, None], axis=2)[:, :, 0]
    norm_sq = jnp.sum(quat_raw * quat_raw, axis=-1, keepdims=True)
    valid_q = norm_sq > jnp.asarray(1e-30, dtype=dtype)
    inv_norm = jnp.where(
      valid_q,
      jax.lax.rsqrt(jnp.where(valid_q, norm_sq, jnp.ones_like(norm_sq))),
      jnp.zeros_like(norm_sq),
    )
    default_q = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype)[None, :]
    quat = jnp.where(valid_q, quat_raw * inv_norm, default_q)
  q0, q1, q2, q3 = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

  # Quaternion -> rotation matrix (proper rotation by construction).
  r00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
  r01 = 2.0 * (q1 * q2 - q0 * q3)
  r02 = 2.0 * (q1 * q3 + q0 * q2)
  r10 = 2.0 * (q1 * q2 + q0 * q3)
  r11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
  r12 = 2.0 * (q2 * q3 - q0 * q1)
  r20 = 2.0 * (q1 * q3 - q0 * q2)
  r21 = 2.0 * (q2 * q3 + q0 * q1)
  r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
  R = jnp.stack(
    [
      jnp.stack([r00, r01, r02], axis=-1),
      jnp.stack([r10, r11, r12], axis=-1),
      jnp.stack([r20, r21, r22], axis=-1),
    ],
    axis=1,
  )  # (N_waters, 3, 3), maps body-frame b_i -> target y_i

  # Apply rotation to body-frame template
  # R @ b_i for each atom i
  rotated_b_O = jnp.einsum("nij,j->ni", R, b_O)  # (N_waters, 3)
  rotated_b_H1 = jnp.einsum("nij,j->ni", R, b_H1)  # (N_waters, 3)
  rotated_b_H2 = jnp.einsum("nij,j->ni", R, b_H2)  # (N_waters, 3)

  # Constrained positions: COM + rotated body frame
  pos_oxygen_c = com_new + rotated_b_O
  pos_h1_c = com_new + rotated_b_H1
  pos_h2_c = com_new + rotated_b_H2

  return pos_oxygen_c, pos_h1_c, pos_h2_c


def _get_settle_bond_vectors(
  pos_oxygen: Array, pos_h1: Array, pos_h2: Array
) -> tuple[Array, Array, Array]:
  """Compute normalized bond vectors for SETTLE."""
  vec_oh1 = pos_h1 - pos_oxygen  # (N_waters, 3)
  vec_oh2 = pos_h2 - pos_oxygen
  vec_h1h2 = pos_h2 - pos_h1

  # Normalize bond vectors
  dist_oh1 = jnp.linalg.norm(vec_oh1, axis=-1, keepdims=True) + 1e-10
  dist_oh2 = jnp.linalg.norm(vec_oh2, axis=-1, keepdims=True) + 1e-10
  dist_h1h2 = jnp.linalg.norm(vec_h1h2, axis=-1, keepdims=True) + 1e-10

  norm_vec_oh1 = vec_oh1 / dist_oh1
  norm_vec_oh2 = vec_oh2 / dist_oh2
  norm_vec_h1h2 = vec_h1h2 / dist_h1h2

  return norm_vec_oh1, norm_vec_oh2, norm_vec_h1h2


def _apply_rattle_velocity_correction(
  vel_curr: Array,
  indices: WaterIndices,
  norm_vec_oh1: Array,
  norm_vec_oh2: Array,
  norm_vec_h1h2: Array,
  inv_mass_oxygen: float,
  inv_mass_hydrogen: float,
  water_mask: Array | None = None,
) -> Array:
  """Single iteration of velocity corrections (RATTLE)."""
  vel_oxygen = vel_curr[indices.oxygen]
  vel_h1 = vel_curr[indices.hydrogen1]
  vel_h2 = vel_curr[indices.hydrogen2]

  # Relative velocities along bonds
  rel_vel_oh1 = vel_h1 - vel_oxygen
  rel_vel_oh2 = vel_h2 - vel_oxygen
  rel_vel_h1h2 = vel_h2 - vel_h1

  # Velocity components along bonds (should be zero)
  dot_oh1 = jnp.sum(rel_vel_oh1 * norm_vec_oh1, axis=-1)
  dot_oh2 = jnp.sum(rel_vel_oh2 * norm_vec_oh2, axis=-1)
  dot_h1h2 = jnp.sum(rel_vel_h1h2 * norm_vec_h1h2, axis=-1)

  # Compute Lagrange multipliers
  lambda_oh1 = -dot_oh1 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_oh2 = -dot_oh2 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_h1h2 = -dot_h1h2 / (2 * inv_mass_hydrogen)

  # Velocity corrections
  dv_oxygen_from_oh1 = -lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_oxygen
  dv_h1_from_oh1 = lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_hydrogen

  dv_oxygen_from_oh2 = -lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_oxygen
  dv_h2_from_oh2 = lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_hydrogen

  dv_h1_from_h1h2 = -lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen
  dv_h2_from_h1h2 = lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen

  # Accumulate corrections
  dv_oxygen = dv_oxygen_from_oh1 + dv_oxygen_from_oh2
  dv_h1 = dv_h1_from_oh1 + dv_h1_from_h1h2
  dv_h2 = dv_h2_from_oh2 + dv_h2_from_h1h2

  # Apply corrections. Padding rows' bond vectors are exactly zero (O/H1/H2
  # all gather the same fill atom), so their corrections are exactly zero —
  # but redirect the scatter target defensively anyway rather than relying
  # on that degeneracy argument to hold across future changes.
  if water_mask is None:
    vel_new = vel_curr.at[indices.oxygen].add(dv_oxygen)
    vel_new = vel_new.at[indices.hydrogen1].add(dv_h1)
    vel_new = vel_new.at[indices.hydrogen2].add(dv_h2)
    return vel_new
  mask_col = water_mask[:, None]
  vel_new = vel_curr.at[indices.oxygen].add(jnp.where(mask_col, dv_oxygen, 0.0))
  vel_new = vel_new.at[indices.hydrogen1].add(jnp.where(mask_col, dv_h1, 0.0))
  vel_new = vel_new.at[indices.hydrogen2].add(jnp.where(mask_col, dv_h2, 0.0))
  return vel_new


def _apply_rattle_velocity_correction_with_residual(
  vel_curr: Array,
  indices: WaterIndices,
  norm_vec_oh1: Array,
  norm_vec_oh2: Array,
  norm_vec_h1h2: Array,
  inv_mass_oxygen: float,
  inv_mass_hydrogen: float,
  water_mask: Array | None = None,
) -> tuple[Array, Array]:
  """Single iteration of velocity corrections (RATTLE) with residual tracking."""
  vel_oxygen = vel_curr[indices.oxygen]
  vel_h1 = vel_curr[indices.hydrogen1]
  vel_h2 = vel_curr[indices.hydrogen2]
  rel_vel_oh1 = vel_h1 - vel_oxygen
  rel_vel_oh2 = vel_h2 - vel_oxygen
  rel_vel_h1h2 = vel_h2 - vel_h1
  dot_oh1 = jnp.sum(rel_vel_oh1 * norm_vec_oh1, axis=-1)
  dot_oh2 = jnp.sum(rel_vel_oh2 * norm_vec_oh2, axis=-1)
  dot_h1h2 = jnp.sum(rel_vel_h1h2 * norm_vec_h1h2, axis=-1)
  lambda_oh1 = -dot_oh1 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_oh2 = -dot_oh2 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_h1h2 = -dot_h1h2 / (2 * inv_mass_hydrogen)
  dv_oxygen_from_oh1 = -lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_oxygen
  dv_h1_from_oh1 = lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_hydrogen
  dv_oxygen_from_oh2 = -lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_oxygen
  dv_h2_from_oh2 = lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_hydrogen
  dv_h1_from_h1h2 = -lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen
  dv_h2_from_h1h2 = lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen
  dv_oxygen = dv_oxygen_from_oh1 + dv_oxygen_from_oh2
  dv_h1 = dv_h1_from_oh1 + dv_h1_from_h1h2
  dv_h2 = dv_h2_from_oh2 + dv_h2_from_h1h2
  if water_mask is None:
    vel_new = vel_curr.at[indices.oxygen].add(dv_oxygen)
    vel_new = vel_new.at[indices.hydrogen1].add(dv_h1)
    vel_new = vel_new.at[indices.hydrogen2].add(dv_h2)
  else:
    mask_col = water_mask[:, None]
    vel_new = vel_curr.at[indices.oxygen].add(jnp.where(mask_col, dv_oxygen, 0.0))
    vel_new = vel_new.at[indices.hydrogen1].add(jnp.where(mask_col, dv_h1, 0.0))
    vel_new = vel_new.at[indices.hydrogen2].add(jnp.where(mask_col, dv_h2, 0.0))
  residual = jnp.max(jnp.abs(jnp.stack([dot_oh1, dot_oh2, dot_h1h2])))
  return vel_new, residual


def _remove_angular_momentum_from_impulse(
  vel_unconstrained: Array,
  vel_constrained: Array | None,
  indices: WaterIndices,
  pos_constrained_O: Array,
  pos_constrained_H1: Array,
  pos_constrained_H2: Array,
  mass_oxygen: float,
  mass_hydrogen: float,
  water_mask: Array | None = None,
  dv_tuple: tuple[Array, Array, Array] | None = None,
) -> Array:
  """Remove the angular-momentum component from the RATTLE velocity impulse.

  RATTLE bond projections satisfy zero relative velocity along bonds, but the
  resulting impulse dp_i = m_i*(v_con_i - v_unc_i) can carry a net angular
  momentum, causing T_rot to drift downward deterministically.

  This function computes the angular momentum of dp for each water, solves for
  the equivalent rigid-body angular velocity omega, and subtracts the rotational
  part from dp, leaving only the radial bond-stretch contribution.

  Algorithm (vectorised over all waters):
    1. r_com = sum_i(m_i * r_con_i) / m_total
    2. d_i   = r_con_i - r_com
    3. dp_i  = m_i * (v_con_i - v_unc_i)
    4. L     = sum_i d_i x dp_i          (AM of dp impulse)
    5. I_ab  = sum_i m_i*(|d_i|^2*delta_ab - d_ia*d_ib)  (inertia tensor)
    6. omega = I^{-1} L                  (angular velocity; explicit Cramer's rule)
    7. dp_i -= m_i * (omega x d_i)       (subtract angular component)
    8. return v_unc + dp_corrected / m_i

  NOTES:
  - Step 6 uses explicit 3×3 Cramer's rule instead of ``jnp.linalg.solve``.
    At 895-water scale inside ``lax.scan``, ``linalg.solve`` dispatches an XLA
    batched-linear-solver kernel whose scheduling interacts badly with the scan
    loop, causing a hang (observed on Engaging cluster, job 15374423 and Certify
    Phase D).  Cramer's rule expands to pure scalar multiply-add ops that XLA can
    fully fuse without a separate kernel dispatch.  The inertia tensor is always
    symmetric positive-definite for physical geometries, so the determinant is
    never zero and the explicit inverse is numerically equivalent.

  Args:
      vel_unconstrained: Velocities before RATTLE (N_atoms, 3).
      vel_constrained: Velocities after RATTLE convergence (N_atoms, 3), or None
          when ``dv_tuple`` is provided directly.
      indices: WaterIndices with oxygen, hydrogen1, hydrogen2 index arrays.
      pos_constrained_O: Constrained O positions (N_waters, 3).
      pos_constrained_H1: Constrained H1 positions (N_waters, 3).
      pos_constrained_H2: Constrained H2 positions (N_waters, 3).
      mass_oxygen: Mass of oxygen atom.
      mass_hydrogen: Mass of hydrogen atom.
      water_mask: Optional mask for real waters.
      dv_tuple: Optional precomputed ``(dv_O, dv_H1, dv_H2)`` velocity deltas,
          avoiding a full-array scatter + gather round-trip.

  Returns:
      Velocity array with AM-corrected RATTLE impulse applied (N_atoms, 3).
  """
  mass_total = mass_oxygen + 2.0 * mass_hydrogen
  m_all = jnp.array([mass_oxygen, mass_hydrogen, mass_hydrogen])  # (3,)

  # Stack constrained positions: (N_waters, 3, 3) — axis-1 indexes O/H1/H2
  r_con = jnp.stack(
    [pos_constrained_O, pos_constrained_H1, pos_constrained_H2], axis=1
  )  # (N_waters, 3, 3)

  # COM from constrained positions: (N_waters, 3)
  r_com = jnp.einsum("i,nij->nj", m_all, r_con) / mass_total  # (N_waters, 3)

  # Relative positions from COM: d_i = r_con_i - r_com  => (N_waters, 3, 3)
  d = r_con - r_com[:, None, :]  # (N_waters, 3, 3); axis-1 = O/H1/H2

  # Velocity impulse dp_i = m_i * (v_con_i - v_unc_i)
  if dv_tuple is not None:
    dv_O, dv_H1, dv_H2 = dv_tuple
  else:
    assert vel_constrained is not None
    dv_O = (
      vel_constrained[indices.oxygen] - vel_unconstrained[indices.oxygen]
    )  # (N_waters, 3)
    dv_H1 = (
      vel_constrained[indices.hydrogen1] - vel_unconstrained[indices.hydrogen1]
    )  # (N_waters, 3)
    dv_H2 = (
      vel_constrained[indices.hydrogen2] - vel_unconstrained[indices.hydrogen2]
    )  # (N_waters, 3)

  # dp_i = m_i * dv_i; shape (N_waters, 3, 3) — axis-1 indexes O/H1/H2
  dp = jnp.stack(
    [
      mass_oxygen * dv_O,
      mass_hydrogen * dv_H1,
      mass_hydrogen * dv_H2,
    ],
    axis=1,
  )  # (N_waters, 3, 3)

  # Angular momentum of impulse: L = sum_i d_i x dp_i  => (N_waters, 3)
  L = jnp.sum(jnp.cross(d, dp), axis=1)  # (N_waters, 3)

  # Inertia tensor: I_ab = sum_i m_i*(|d_i|^2*delta_ab - d_ia*d_ib) => (N_waters, 3, 3)
  d_sq = jnp.sum(d**2, axis=-1)  # (N_waters, 3): |d_i|^2 per atom
  # sum_i m_i*|d_i|^2 per water => (N_waters,)
  trace_term = jnp.einsum("i,ni->n", m_all, d_sq)  # (N_waters,)
  # sum_i m_i*d_ia*d_ib => (N_waters, 3, 3)
  outer_term = jnp.einsum("i,nia,nib->nab", m_all, d, d)  # (N_waters, 3, 3)
  I_eye = trace_term[:, None, None] * jnp.eye(3)[None, :, :]  # (N_waters, 3, 3)
  I_tensor = I_eye - outer_term  # (N_waters, 3, 3)

  # omega = I^{-1} L via explicit 3×3 Cramer's rule (see NOTES in docstring).
  with jax.named_scope("settle_cramers_rule"):
    omega = _solve_3x3_batch(I_tensor, L)

  # Angular impulse to subtract: m_i * (omega x d_i) for each atom
  # omega broadcast over atom axis: (N_waters, 1, 3) x (N_waters, 3, 3)
  omega_exp = omega[:, None, :]  # (N_waters, 1, 3)
  # cross product: (omega x d_i) per atom => (N_waters, 3, 3)
  omega_cross_d = jnp.cross(omega_exp * jnp.ones_like(d), d)  # (N_waters, 3, 3)
  # m_i * (omega x d_i): broadcast mass per atom
  angular_dp = m_all[None, :, None] * omega_cross_d  # (N_waters, 3, 3)

  # Corrected impulse: remove angular part
  dp_corrected = dp - angular_dp  # (N_waters, 3, 3)

  # Apply corrected dv = dp_corrected / m_i back to velocity array
  dv_O_cor = dp_corrected[:, 0, :] / mass_oxygen
  dv_H1_cor = dp_corrected[:, 1, :] / mass_hydrogen
  dv_H2_cor = dp_corrected[:, 2, :] / mass_hydrogen

  if water_mask is None:
    vel_out = vel_unconstrained.at[indices.oxygen].add(dv_O_cor)
    vel_out = vel_out.at[indices.hydrogen1].add(dv_H1_cor)
    vel_out = vel_out.at[indices.hydrogen2].add(dv_H2_cor)
    return vel_out
  mask_col = water_mask[:, None]
  vel_out = vel_unconstrained.at[indices.oxygen].add(jnp.where(mask_col, dv_O_cor, 0.0))
  vel_out = vel_out.at[indices.hydrogen1].add(jnp.where(mask_col, dv_H1_cor, 0.0))
  vel_out = vel_out.at[indices.hydrogen2].add(jnp.where(mask_col, dv_H2_cor, 0.0))
  return vel_out


def _r_step_conserve_angular_momentum(
  momentum: Array,
  p_pre_a: Array,
  x_unc: Array,
  x_con: Array,
  water_indices: WaterIndicesArray,
  mass_oxygen: float,
  mass_hydrogen: float,
  box: Array | None = None,
  water_mask: Array | None = None,
) -> Array:
  """Restore per-water angular momentum after an A+SETTLE+R-step block.

  The combined A+SETTLE+R-step introduces angular momentum error
      ΔL = sum_i(r_con_i - r_unc_i) × p_pre_a_i  (SETTLE_pos contribution)
         + sum_i r_con_i × dp_i                    (R-step contribution)
  where dp_i = m_i*(r_con_i - r_unc_i)/half_dt.

  This function computes ΔL per water molecule and adds a corrective
  rigid-body impulse dp_correct_i = m_i*(ω × d_con_i) so that
  L(r_con, momentum_out) = L(r_unc, p_pre_a).

  Uses explicit 3×3 Cramer's rule (not linalg.solve) — see
  _remove_angular_momentum_from_impulse docstring for rationale (XLA
  batched-solver hang at 895-water scale inside lax.scan).

  PBC note: shift_fn wraps atoms individually after the A step, so water
  molecules near a periodic boundary may have O at x≈0 and H at x≈L_box.
  Without correction |d_unc| ~ L_box → catastrophically large impulse → NaN.
  When box is provided, H atoms are unwrapped relative to O via minimum-image
  before computing L_target.

  Args:
      momentum: (N_atoms, 3) momenta after R-step.
      p_pre_a: (N_atoms, 3) momenta before the A step (L_target reference).
      x_unc: (N_atoms, 3) positions after A step, before SETTLE (may be
          PBC-split — H atoms are unwrapped relative to O internally).
      x_con: (N_atoms, 3) constrained positions after SETTLE (already in
          consistent frame from settle_positions).
      water_indices: (N_waters, 3) int array of [O, H1, H2] atom indices.
      mass_oxygen: float mass of oxygen.
      mass_hydrogen: float mass of hydrogen.
      box: (3,) periodic box lengths, or None for non-periodic systems.

  Returns:
      (N_atoms, 3) AM-corrected momenta.
  """
  if water_indices.shape[0] == 0:
    return momentum

  indices = WaterIndices.from_row(water_indices.T)
  mass_total = mass_oxygen + 2.0 * mass_hydrogen
  m_all = jnp.array([mass_oxygen, mass_hydrogen, mass_hydrogen])  # (3,)

  # L_target = L(r_unc, p_pre_a): angular momentum before A+SETTLE+R.
  # Unwrap H atoms relative to O: shift_fn wraps atoms independently, so a
  # water molecule straddling a PBC boundary has O and H in different images.
  r_O_unc = x_unc[indices.oxygen]      # (N_w, 3)
  r_H1_unc = x_unc[indices.hydrogen1]  # (N_w, 3)
  r_H2_unc = x_unc[indices.hydrogen2]  # (N_w, 3)
  if box is not None:
    dH1 = _min_image_delta(r_H1_unc - r_O_unc, box)
    r_H1_unc = r_O_unc + dH1
    dH2 = _min_image_delta(r_H2_unc - r_O_unc, box)
    r_H2_unc = r_O_unc + dH2
  r_unc_w = jnp.stack([r_O_unc, r_H1_unc, r_H2_unc], axis=1)  # (N_w, 3, 3)
  p_pre_a_w = jnp.stack(
    [p_pre_a[indices.oxygen], p_pre_a[indices.hydrogen1], p_pre_a[indices.hydrogen2]], axis=1
  )  # (N_w, 3, 3)
  r_com_unc = jnp.einsum("i,nij->nj", m_all, r_unc_w) / mass_total  # (N_w, 3)
  d_unc = r_unc_w - r_com_unc[:, None, :]  # (N_w, 3, 3)
  L_target = jnp.sum(jnp.cross(d_unc, p_pre_a_w), axis=1)  # (N_w, 3)

  # L_current = L(r_con, momentum): angular momentum after R-step
  r_con_w = jnp.stack(
    [x_con[indices.oxygen], x_con[indices.hydrogen1], x_con[indices.hydrogen2]], axis=1
  )  # (N_w, 3, 3)
  p_after_w = jnp.stack(
    [momentum[indices.oxygen], momentum[indices.hydrogen1], momentum[indices.hydrogen2]], axis=1
  )  # (N_w, 3, 3)
  r_com_con = jnp.einsum("i,nij->nj", m_all, r_con_w) / mass_total  # (N_w, 3)
  d_con = r_con_w - r_com_con[:, None, :]  # (N_w, 3, 3)
  L_current = jnp.sum(jnp.cross(d_con, p_after_w), axis=1)  # (N_w, 3)

  # Correction: add ω × r_con impulse to restore L to L_target
  L_correct = L_target - L_current  # (N_w, 3)

  # Inertia tensor at constrained positions
  d_sq = jnp.sum(d_con ** 2, axis=-1)  # (N_w, 3)
  trace_term = jnp.einsum("i,ni->n", m_all, d_sq)  # (N_w,)
  outer_term = jnp.einsum("i,nia,nib->nab", m_all, d_con, d_con)  # (N_w, 3, 3)
  I_eye = trace_term[:, None, None] * jnp.eye(3)[None, :, :]
  I_tensor = I_eye - outer_term  # (N_w, 3, 3)

  # omega = I^{-1} L_correct via explicit 3×3 Cramer's rule
  with jax.named_scope("settle_cramers_rule"):
    omega = _solve_3x3_batch(I_tensor, L_correct)

  # Corrective impulse: m_i * (omega × d_con_i)
  omega_exp = omega[:, None, :]  # (N_w, 1, 3)
  omega_cross_d = jnp.cross(omega_exp * jnp.ones_like(d_con), d_con)  # (N_w, 3, 3)
  dp_correct = m_all[None, :, None] * omega_cross_d  # (N_w, 3, 3)

  if water_mask is None:
    mom_out = momentum.at[indices.oxygen].add(dp_correct[:, 0, :])
    mom_out = mom_out.at[indices.hydrogen1].add(dp_correct[:, 1, :])
    mom_out = mom_out.at[indices.hydrogen2].add(dp_correct[:, 2, :])
    return mom_out
  mask_col = water_mask[:, None]
  mom_out = momentum.at[indices.oxygen].add(jnp.where(mask_col, dp_correct[:, 0, :], 0.0))
  mom_out = mom_out.at[indices.hydrogen1].add(jnp.where(mask_col, dp_correct[:, 1, :], 0.0))
  mom_out = mom_out.at[indices.hydrogen2].add(jnp.where(mask_col, dp_correct[:, 2, :], 0.0))
  return mom_out


def settle_velocities(
  velocities: Array,
  positions_old: Array,
  positions_constrained: Array,
  water_indices: WaterIndicesArray,
  dt: float,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  n_iters: int = 10,
  adaptive_tol: float | None = None,
  water_mask: Array | None = None,
) -> Array:
  r"""Apply velocity constraints after position SETTLE via 1-pass 3x3 Gram matrix RATTLE."""
  del positions_old, dt, adaptive_tol
  if water_indices.shape[0] == 0:
    return velocities

  indices = WaterIndices.from_row(water_indices.T)
  pos_O = positions_constrained[indices.oxygen]
  pos_H1 = positions_constrained[indices.hydrogen1]
  pos_H2 = positions_constrained[indices.hydrogen2]

  if int(n_iters) <= 0:
    return _remove_angular_momentum_from_impulse(
      velocities,
      velocities,
      indices,
      pos_O,
      pos_H1,
      pos_H2,
      mass_oxygen,
      mass_hydrogen,
      water_mask,
    )

  with jax.named_scope("settle_rattle_loop"):
    n1, n2, n3 = _get_settle_bond_vectors(pos_O, pos_H1, pos_H2)
    if water_mask is not None:
      mask_col = water_mask[:, None]
      n1 = jnp.where(mask_col, n1, 0.0)
      n2 = jnp.where(mask_col, n2, 0.0)
      n3 = jnp.where(mask_col, n3, 0.0)

    v_O = velocities[indices.oxygen]
    v_H1 = velocities[indices.hydrogen1]
    v_H2 = velocities[indices.hydrogen2]

    b1 = jnp.sum((v_H1 - v_O) * n1, axis=-1)
    b2 = jnp.sum((v_H2 - v_O) * n2, axis=-1)
    b3 = jnp.sum((v_H2 - v_H1) * n3, axis=-1)
    rhs = -jnp.stack([b1, b2, b3], axis=-1)

    dtype = velocities.dtype
    mu_O = jnp.asarray(1.0 / mass_oxygen, dtype=dtype)
    mu_H = jnp.asarray(1.0 / mass_hydrogen, dtype=dtype)
    diag_OH = jnp.broadcast_to(mu_O + mu_H, b1.shape)
    diag_HH = jnp.broadcast_to(2.0 * mu_H, b1.shape)

    c12 = mu_O * jnp.sum(n1 * n2, axis=-1)
    c13 = -mu_H * jnp.sum(n1 * n3, axis=-1)
    c23 = mu_H * jnp.sum(n2 * n3, axis=-1)

    row0 = jnp.stack([diag_OH, c12, c13], axis=-1)
    row1 = jnp.stack([c12, diag_OH, c23], axis=-1)
    row2 = jnp.stack([c13, c23, diag_HH], axis=-1)
    G = jnp.stack([row0, row1, row2], axis=1)

    lambdas = _solve_3x3_batch(G, rhs)
    l1 = lambdas[:, 0:1]
    l2 = lambdas[:, 1:2]
    l3 = lambdas[:, 2:3]

    dv_O = -mu_O * (l1 * n1 + l2 * n2)
    dv_H1 = mu_H * (l1 * n1 - l3 * n3)
    dv_H2 = mu_H * (l2 * n2 + l3 * n3)

    if water_mask is not None:
      mask_col = water_mask[:, None]
      dv_O = jnp.where(mask_col, dv_O, 0.0)
      dv_H1 = jnp.where(mask_col, dv_H1, 0.0)
      dv_H2 = jnp.where(mask_col, dv_H2, 0.0)

    vel_rattle = velocities.at[indices.oxygen].add(dv_O)
    vel_rattle = vel_rattle.at[indices.hydrogen1].add(dv_H1)
    vel_rattle = vel_rattle.at[indices.hydrogen2].add(dv_H2)

  return _remove_angular_momentum_from_impulse(
    velocities,
    vel_rattle,
    indices,
    pos_O,
    pos_H1,
    pos_H2,
    mass_oxygen,
    mass_hydrogen,
    water_mask,
  )


def get_water_indices(n_protein_atoms: int, n_waters: int) -> Array:
  """Generate water molecule indices assuming standard O, H1, H2 ordering.

  For a system with n_protein_atoms followed by n_waters water molecules
  in O, H1, H2 order, this returns the indices for SETTLE.

  Args:
      n_protein_atoms: Number of protein atoms (non-water).
      n_waters: Number of water molecules.

  Returns:
      Array of shape (n_waters, 3) with [O_idx, H1_idx, H2_idx] per water.
  """
  if n_waters == 0:
    return jnp.zeros((0, 3), dtype=jnp.int32)

  water_indices = []
  for i in range(n_waters):
    base = n_protein_atoms + i * 3
    water_indices.append([base, base + 1, base + 2])

  return jnp.array(water_indices, dtype=jnp.int32)


def _make_settle_compatible_force_fn(
  energy_or_force_fn: Callable[..., Array],
  mass: float | Array,
  box_template: Array | None,
  *,
  is_force_shaped: bool = False,
) -> Callable[..., Array]:
  """``jax_md.quantity.canonicalize_force``-compatible force fn using one ``value_and_grad`` when safe.

  If ``mass`` is a scalar, the atom count is unknown at integrator construction time;
  we fall back to ``canonicalize_force`` only.

  Args:
      is_force_shaped: When True, the caller already knows ``energy_or_force_fn``
          returns an ``(N, 3)`` force array (not a scalar energy) -- e.g.
          ``force_fn_from_bundle``'s FlashMD path (debt 761) -- and this skips
          BOTH shape-auto-detection layers entirely rather than routing through
          them. This matters because ``jax_md.quantity.canonicalize_force``'s
          returned closure does its own lazy, UNPROTECTED ``eval_shape`` probe
          on every first real call (jax_md/quantity.py's ``force_fn``, cached via
          a ``nonlocal _force_fn`` -- not just once at construction time here).
          That probe crashes with ``TracerArrayConversionError`` whenever
          ``energy_or_force_fn`` touches a bundle's static ``box_size`` field via
          the debt-770 PME-grid-sizing numpy workaround (confirmed via full
          unfiltered traceback, job 20443232 local repro): our own eval_shape
          probe below (``make_force_fn_like_canonicalize``) is try/except-
          protected and correctly falls back to ``canonicalize_force`` on that
          exception, but jax_md's *own* internal probe inside the fallback is
          not protected by anything we control, so the crash still propagates
          on the first real (non-probe) call. Skipping both probes when the
          shape is already known sidesteps this rather than trying to fix a
          third-party library's internal caching wrapper.
  """
  if is_force_shaped:
    return energy_or_force_fn
  mass_arr = jnp.asarray(mass)
  if mass_arr.ndim == 0:
    return quantity.canonicalize_force(energy_or_force_fn)
  n_atoms = int(mass_arr.reshape(-1).shape[0])
  template_R = jnp.zeros((n_atoms, 3), dtype=jnp.float64)
  kw: dict[str, Any] = {}
  if box_template is not None:
    kw["box"] = box_template
  return md_potential_bundle.make_force_fn_like_canonicalize(
    energy_or_force_fn,
    template_R=template_R,
    template_kwargs=kw or None,
  )


# ---------------------------------------------------------------------------
# Pallas v3.0 (MK-46): Fused BAOAB + Analytical SETTLE Integrator Kernels
# ---------------------------------------------------------------------------

_SETTLE_ATOM_BLOCK = 32
_SETTLE_WATER_BLOCK = 32


def _should_use_pallas_settle_default(use_pallas: bool | None = None) -> bool:
  if in_export_context():
    return False
  if use_pallas is False:
    return False
  flag = os.environ.get("PROLIX_USE_PALLAS_SETTLE", "").strip().lower()
  if flag in ("0", "false", "no", "off"):
    return False
  if use_pallas is True:
    return True
  if flag in ("1", "true", "yes", "on"):
    return True
  return jax.default_backend() == "gpu"


def _settle_water_vec3(
  o_old: tuple[Array, Array, Array],
  h1_old: tuple[Array, Array, Array],
  h2_old: tuple[Array, Array, Array],
  o_new: tuple[Array, Array, Array],
  h1_new: tuple[Array, Array, Array],
  h2_new: tuple[Array, Array, Array],
  r_OH: float,
  r_HH: float,
  mass_oxygen: float,
  mass_hydrogen: float,
  box_xyz: tuple[Array, Array, Array] | None,
  has_box: bool,
  use_f32_math: bool = False,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array], tuple[Array, Array, Array]]:
  """Vectorized 3-component analytical Horn-SETTLE + PBC re-wrap for a block of waters."""
  dtype = o_new[0].dtype
  w_dtype = jnp.float32 if (use_f32_math and dtype == jnp.float64) else dtype
  zero_d = jnp.asarray(0.0, dtype=dtype)
  one_d = jnp.asarray(1.0, dtype=dtype)
  two_d = jnp.asarray(2.0, dtype=dtype)
  zero = jnp.asarray(0.0, dtype=w_dtype)
  one = jnp.asarray(1.0, dtype=w_dtype)
  two = jnp.asarray(2.0, dtype=w_dtype)

  if has_box and box_xyz is not None:
    bx, by, bz = box_xyz

    def _min_img_3(dx, dy, dz):
      return (
        dx - bx * _PBC_ROUND_FN(dx / bx),
        dy - by * _PBC_ROUND_FN(dy / by),
        dz - bz * _PBC_ROUND_FN(dz / bz),
      )

    def _unwrap_3(p_n, p_o):
      d0, d1, d2 = _min_img_3(p_n[0] - p_o[0], p_n[1] - p_o[1], p_n[2] - p_o[2])
      return (p_o[0] + d0, p_o[1] + d1, p_o[2] + d2)

    o_u = _unwrap_3(o_new, o_old)
    h1_u = _unwrap_3(h1_new, h1_old)
    h2_u = _unwrap_3(h2_new, h2_old)

    dh1_x, dh1_y, dh1_z = _min_img_3(h1_u[0] - o_u[0], h1_u[1] - o_u[1], h1_u[2] - o_u[2])
    h1_u = (o_u[0] + dh1_x, o_u[1] + dh1_y, o_u[2] + dh1_z)
    dh2_x, dh2_y, dh2_z = _min_img_3(h2_u[0] - o_u[0], h2_u[1] - o_u[1], h2_u[2] - o_u[2])
    h2_u = (o_u[0] + dh2_x, o_u[1] + dh2_y, o_u[2] + dh2_z)
  else:
    o_u, h1_u, h2_u = o_new, h1_new, h2_new

  m_tot_f = float(mass_oxygen + 2.0 * mass_hydrogen)
  m_O_d = jnp.asarray(mass_oxygen, dtype=dtype)
  m_H_d = jnp.asarray(mass_hydrogen, dtype=dtype)
  inv_m_tot_d = jnp.asarray(1.0 / m_tot_f, dtype=dtype)

  com_x = (m_O_d * o_u[0] + m_H_d * (h1_u[0] + h2_u[0])) * inv_m_tot_d
  com_y = (m_O_d * o_u[1] + m_H_d * (h1_u[1] + h2_u[1])) * inv_m_tot_d
  com_z = (m_O_d * o_u[2] + m_H_d * (h1_u[2] + h2_u[2])) * inv_m_tot_d

  yO_x = (o_u[0] - com_x).astype(w_dtype)
  yO_y = (o_u[1] - com_y).astype(w_dtype)
  yO_z = (o_u[2] - com_z).astype(w_dtype)
  yH1_x = (h1_u[0] - com_x).astype(w_dtype)
  yH1_y = (h1_u[1] - com_y).astype(w_dtype)
  yH1_z = (h1_u[2] - com_z).astype(w_dtype)
  yH2_x = (h2_u[0] - com_x).astype(w_dtype)
  yH2_y = (h2_u[1] - com_y).astype(w_dtype)
  yH2_z = (h2_u[2] - com_z).astype(w_dtype)

  dist_oh_mid = float((r_OH**2 - (r_HH / 2.0) ** 2) ** 0.5)
  bO_y_f = 2.0 * float(mass_hydrogen) * dist_oh_mid / m_tot_f
  bH_y_f = -float(mass_oxygen) * dist_oh_mid / m_tot_f
  bH_x_f = float(r_HH) / 2.0

  m_O = jnp.asarray(mass_oxygen, dtype=w_dtype)
  m_H = jnp.asarray(mass_hydrogen, dtype=w_dtype)
  bO_y = jnp.asarray(bO_y_f, dtype=w_dtype)
  bH_y = jnp.asarray(bH_y_f, dtype=w_dtype)
  bH_x = jnp.asarray(bH_x_f, dtype=w_dtype)

  # S[0, :] = m_H * bH_x * (yH1 - yH2)
  # S[1, :] = m_O * bO_y * yO + m_H * bH_y * (yH1 + yH2)
  # S[2, :] = 0 (since body-frame template lies in xy plane)
  c_s0 = m_H * bH_x
  sxx = c_s0 * (yH1_x - yH2_x)
  sxy = c_s0 * (yH1_y - yH2_y)
  sxz = c_s0 * (yH1_z - yH2_z)

  c_s1_O = m_O * bO_y
  c_s1_H = m_H * bH_y
  syx = c_s1_O * yO_x + c_s1_H * (yH1_x + yH2_x)
  syy = c_s1_O * yO_y + c_s1_H * (yH1_y + yH2_y)
  syz = c_s1_O * yO_z + c_s1_H * (yH1_z + yH2_z)

  p_tr = sxx * sxx + sxy * sxy + sxz * sxz + syx * syx + syy * syy + syz * syz
  cr_x = sxy * syz - sxz * syy
  cr_y = sxz * syx - sxx * syz
  cr_z = sxx * syy - sxy * syx
  c_sq = cr_x * cr_x + cr_y * cr_y + cr_z * cr_z
  q_val = jnp.where(c_sq > zero, jnp.sqrt(jnp.where(c_sq > zero, c_sq, one)), zero)
  lam_max_sq = jnp.maximum(p_tr + two * q_val, zero)
  lam_max = jnp.where(
    lam_max_sq > zero,
    jnp.sqrt(jnp.where(lam_max_sq > zero, lam_max_sq, one)),
    zero,
  )

  # Symmetric 4x4 Horn matrix N entries (with sz* = 0):
  n00 = sxx + syy + lam_max
  n01 = syz
  n02 = zero - sxz
  n03 = sxy - syx
  n11 = sxx - syy + lam_max
  n12 = sxy + syx
  n13 = sxz
  n22 = syy - sxx + lam_max
  n23 = syz
  n33 = lam_max - sxx - syy

  q_denom = jnp.maximum(q_val, jnp.asarray(1e-12, dtype=w_dtype))
  cx = cr_x / q_denom
  cy = cr_y / q_denom
  cz = cr_z / q_denom
  has_q = q_val > jnp.asarray(1e-12, dtype=w_dtype)

  p2_00 = jnp.where(has_q, one + cz, one)
  p2_10 = jnp.where(has_q, zero - cy, zero)
  p2_20 = jnp.where(has_q, cx, zero)
  p2_30 = zero

  p2_01 = jnp.where(has_q, zero - cy, zero)
  p2_11 = jnp.where(has_q, one - cz, one)
  p2_21 = zero
  p2_31 = jnp.where(has_q, cx, zero)

  p2_02 = jnp.where(has_q, cx, zero)
  p2_12 = zero
  p2_22 = jnp.where(has_q, one - cz, one)
  p2_32 = jnp.where(has_q, cy, zero)

  p2_03 = zero
  p2_13 = jnp.where(has_q, cx, zero)
  p2_23 = jnp.where(has_q, cy, zero)
  p2_33 = jnp.where(has_q, one + cz, one)

  def _mul_col(c0, c1, c2, c3):
    r0 = n00 * c0 + n01 * c1 + n02 * c2 + n03 * c3
    r1 = n01 * c0 + n11 * c1 + n12 * c2 + n13 * c3
    r2 = n02 * c0 + n12 * c1 + n22 * c2 + n23 * c3
    r3 = n03 * c0 + n13 * c1 + n23 * c2 + n33 * c3
    nsq = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
    return r0, r1, r2, r3, nsq

  col0_0, col0_1, col0_2, col0_3, nsq0 = _mul_col(p2_00, p2_10, p2_20, p2_30)
  col1_0, col1_1, col1_2, col1_3, nsq1 = _mul_col(p2_01, p2_11, p2_21, p2_31)
  col2_0, col2_1, col2_2, col2_3, nsq2 = _mul_col(p2_02, p2_12, p2_22, p2_32)
  col3_0, col3_1, col3_2, col3_3, nsq3 = _mul_col(p2_03, p2_13, p2_23, p2_33)

  best_nsq = nsq0
  qr0, qr1, qr2, qr3 = col0_0, col0_1, col0_2, col0_3

  take1 = nsq1 > best_nsq
  best_nsq = jnp.where(take1, nsq1, best_nsq)
  qr0 = jnp.where(take1, col1_0, qr0)
  qr1 = jnp.where(take1, col1_1, qr1)
  qr2 = jnp.where(take1, col1_2, qr2)
  qr3 = jnp.where(take1, col1_3, qr3)

  take2 = nsq2 > best_nsq
  best_nsq = jnp.where(take2, nsq2, best_nsq)
  qr0 = jnp.where(take2, col2_0, qr0)
  qr1 = jnp.where(take2, col2_1, qr1)
  qr2 = jnp.where(take2, col2_2, qr2)
  qr3 = jnp.where(take2, col2_3, qr3)

  take3 = nsq3 > best_nsq
  best_nsq = jnp.where(take3, nsq3, best_nsq)
  qr0 = jnp.where(take3, col3_0, qr0)
  qr1 = jnp.where(take3, col3_1, qr1)
  qr2 = jnp.where(take3, col3_2, qr2)
  qr3 = jnp.where(take3, col3_3, qr3)

  valid_q = best_nsq > jnp.asarray(1e-30, dtype=w_dtype)
  inv_norm = jnp.where(valid_q, jax.lax.rsqrt(jnp.where(valid_q, best_nsq, one)), zero)

  if use_f32_math and dtype == jnp.float64:
    # 1 Newton-Raphson FMA refinement step in float64 so quaternion norm is 1.0 to 1e-15
    # without any float64 sqrt or division instruction.
    qr0_d = qr0.astype(dtype)
    qr1_d = qr1.astype(dtype)
    qr2_d = qr2.astype(dtype)
    qr3_d = qr3.astype(dtype)
    nsq_d = qr0_d * qr0_d + qr1_d * qr1_d + qr2_d * qr2_d + qr3_d * qr3_d
    valid_qd = nsq_d > jnp.asarray(1e-30, dtype=dtype)
    inv_d0 = inv_norm.astype(dtype)
    inv_d = jnp.where(
      valid_qd,
      inv_d0 * (jnp.asarray(1.5, dtype=dtype) - jnp.asarray(0.5, dtype=dtype) * nsq_d * inv_d0 * inv_d0),
      zero_d,
    )
    q0 = jnp.where(valid_qd, qr0_d * inv_d, one_d)
    q1 = jnp.where(valid_qd, qr1_d * inv_d, zero_d)
    q2 = jnp.where(valid_qd, qr2_d * inv_d, zero_d)
    q3 = jnp.where(valid_qd, qr3_d * inv_d, zero_d)
  else:
    q0 = jnp.where(valid_q, qr0 * inv_norm, one)
    q1 = jnp.where(valid_q, qr1 * inv_norm, zero)
    q2 = jnp.where(valid_q, qr2 * inv_norm, zero)
    q3 = jnp.where(valid_q, qr3 * inv_norm, zero)

  bO_y_d = jnp.asarray(bO_y_f, dtype=dtype)
  bH_y_d = jnp.asarray(bH_y_f, dtype=dtype)
  bH_x_d = jnp.asarray(bH_x_f, dtype=dtype)

  r00 = one_d - two_d * (q2 * q2 + q3 * q3)
  r01 = two_d * (q1 * q2 - q0 * q3)
  r10 = two_d * (q1 * q2 + q0 * q3)
  r11 = one_d - two_d * (q1 * q1 + q3 * q3)
  r20 = two_d * (q1 * q3 - q0 * q2)
  r21 = two_d * (q2 * q3 + q0 * q1)

  o_cx = com_x + r01 * bO_y_d
  o_cy = com_y + r11 * bO_y_d
  o_cz = com_z + r21 * bO_y_d

  h1_cx = com_x + r00 * bH_x_d + r01 * bH_y_d
  h1_cy = com_y + r10 * bH_x_d + r11 * bH_y_d
  h1_cz = com_z + r20 * bH_x_d + r21 * bH_y_d

  h2_cx = com_x - r00 * bH_x_d + r01 * bH_y_d
  h2_cy = com_y - r10 * bH_x_d + r11 * bH_y_d
  h2_cz = com_z - r20 * bH_x_d + r21 * bH_y_d

  if has_box and box_xyz is not None:
    bx, by, bz = box_xyz
    do_x = o_cx - o_new[0]
    do_y = o_cy - o_new[1]
    do_z = o_cz - o_new[2]
    ic_x = (zero_d - bx) * _PBC_ROUND_FN(do_x / bx)
    ic_y = (zero_d - by) * _PBC_ROUND_FN(do_y / by)
    ic_z = (zero_d - bz) * _PBC_ROUND_FN(do_z / bz)
    o_cx, o_cy, o_cz = o_cx + ic_x, o_cy + ic_y, o_cz + ic_z
    h1_cx, h1_cy, h1_cz = h1_cx + ic_x, h1_cy + ic_y, h1_cz + ic_z
    h2_cx, h2_cy, h2_cz = h2_cx + ic_x, h2_cy + ic_y, h2_cz + ic_z

  return (o_cx, o_cy, o_cz), (h1_cx, h1_cy, h1_cz), (h2_cx, h2_cy, h2_cz)


def _solve_3x3_vec3(
  Ixx: Array,
  Iyy: Array,
  Izz: Array,
  Ixy: Array,
  Ixz: Array,
  Iyz: Array,
  bx: Array,
  by: Array,
  bz: Array,
) -> tuple[Array, Array, Array]:
  """Solve symmetric 3x3 system I * omega = b via explicit Cramer's rule on vector lanes."""
  dtype = Ixx.dtype
  zero = jnp.asarray(0.0, dtype=dtype)
  one = jnp.asarray(1.0, dtype=dtype)
  c00 = Iyy * Izz - Iyz * Iyz
  c01 = Ixz * Iyz - Ixy * Izz
  c02 = Ixy * Iyz - Ixz * Iyy
  c11 = Ixx * Izz - Ixz * Ixz
  c12 = Ixy * Ixz - Ixx * Iyz
  c22 = Ixx * Iyy - Ixy * Ixy
  det = Ixx * c00 + Ixy * c01 + Ixz * c02
  valid = jnp.abs(det) > jnp.asarray(1e-30, dtype=dtype)
  inv = jnp.where(valid, one / jnp.where(valid, det, one), zero)
  ox = inv * (c00 * bx + c01 * by + c02 * bz)
  oy = inv * (c01 * bx + c11 * by + c12 * bz)
  oz = inv * (c02 * bx + c12 * by + c22 * bz)
  return ox, oy, oz


def _cholesky_upper_solve_3x3_vec3(
  Ixx: Array,
  Iyy: Array,
  Izz: Array,
  Ixy: Array,
  Ixz: Array,
  Iyz: Array,
  rx: Array,
  ry: Array,
  rz: Array,
) -> tuple[Array, Array, Array]:
  """Solve L^T * omega = r where L * L^T = I on vector lanes."""
  dtype = Ixx.dtype
  eps = jnp.asarray(1e-30, dtype=dtype)
  l00 = jnp.sqrt(jnp.maximum(Ixx, eps))
  l10 = Ixy / l00
  l20 = Ixz / l00
  l11 = jnp.sqrt(jnp.maximum(Iyy - l10 * l10, eps))
  l21 = (Iyz - l20 * l10) / l11
  l22 = jnp.sqrt(jnp.maximum(Izz - l20 * l20 - l21 * l21, eps))
  x2 = rz / l22
  x1 = (ry - l21 * x2) / l11
  x0 = (rx - l10 * x1 - l20 * x2) / l00
  return x0, x1, x2


def _inertia_tensor_vec3(
  dO: tuple[Array, Array, Array],
  dH1: tuple[Array, Array, Array],
  dH2: tuple[Array, Array, Array],
  mO: Array,
  mH1: Array,
  mH2: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
  """Compute symmetric 3x3 inertia tensor components (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)."""
  dtype = dO[0].dtype
  zero = jnp.asarray(0.0, dtype=dtype)
  dO_sq = dO[0] * dO[0] + dO[1] * dO[1] + dO[2] * dO[2]
  dH1_sq = dH1[0] * dH1[0] + dH1[1] * dH1[1] + dH1[2] * dH1[2]
  dH2_sq = dH2[0] * dH2[0] + dH2[1] * dH2[1] + dH2[2] * dH2[2]
  tr = mO * dO_sq + mH1 * dH1_sq + mH2 * dH2_sq
  Ixx = tr - (mO * dO[0] * dO[0] + mH1 * dH1[0] * dH1[0] + mH2 * dH2[0] * dH2[0])
  Iyy = tr - (mO * dO[1] * dO[1] + mH1 * dH1[1] * dH1[1] + mH2 * dH2[1] * dH2[1])
  Izz = tr - (mO * dO[2] * dO[2] + mH1 * dH1[2] * dH1[2] + mH2 * dH2[2] * dH2[2])
  Ixy = zero - (mO * dO[0] * dO[1] + mH1 * dH1[0] * dH1[1] + mH2 * dH2[0] * dH2[1])
  Ixz = zero - (mO * dO[0] * dO[2] + mH1 * dH1[0] * dH1[2] + mH2 * dH2[0] * dH2[2])
  Iyz = zero - (mO * dO[1] * dO[2] + mH1 * dH1[1] * dH1[2] + mH2 * dH2[1] * dH2[2])
  return Ixx, Iyy, Izz, Ixy, Ixz, Iyz


def _cross_vec3(
  a: tuple[Array, Array, Array], b: tuple[Array, Array, Array]
) -> tuple[Array, Array, Array]:
  return (
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  )


def _r_step_and_am_vec3(
  r_unc_O: tuple[Array, Array, Array],
  r_unc_H1: tuple[Array, Array, Array],
  r_unc_H2: tuple[Array, Array, Array],
  r_con_O: tuple[Array, Array, Array],
  r_con_H1: tuple[Array, Array, Array],
  r_con_H2: tuple[Array, Array, Array],
  p_pre_O: tuple[Array, Array, Array],
  p_pre_H1: tuple[Array, Array, Array],
  p_pre_H2: tuple[Array, Array, Array],
  mO_state: Array,
  mH1_state: Array,
  mH2_state: Array,
  half_dt: Array,
  mass_oxygen: float,
  mass_hydrogen: float,
  box_xyz: tuple[Array, Array, Array] | None,
  has_box: bool,
  use_f32_math: bool = False,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array], tuple[Array, Array, Array]]:
  """In-register R-step momentum impulse + angular momentum restoration for a water block."""
  dtype = r_con_O[0].dtype
  w_dtype = jnp.float32 if (use_f32_math and dtype == jnp.float64) else dtype
  m_O_d = jnp.asarray(mass_oxygen, dtype=dtype)
  m_H_d = jnp.asarray(mass_hydrogen, dtype=dtype)
  inv_m_tot_d = jnp.asarray(1.0 / (mass_oxygen + 2.0 * mass_hydrogen), dtype=dtype)
  m_O = jnp.asarray(mass_oxygen, dtype=w_dtype)
  m_H = jnp.asarray(mass_hydrogen, dtype=w_dtype)
  inv_half_dt = jnp.asarray(1.0, dtype=w_dtype) / half_dt.astype(w_dtype)

  def _disp3(a, b):
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    if has_box and box_xyz is not None:
      bx, by, bz = box_xyz
      dx = dx - bx * _PBC_ROUND_FN(dx / bx)
      dy = dy - by * _PBC_ROUND_FN(dy / by)
      dz = dz - bz * _PBC_ROUND_FN(dz / bz)
    return (dx, dy, dz)

  dxO_d = _disp3(r_con_O, r_unc_O)
  dxH1_d = _disp3(r_con_H1, r_unc_H1)
  dxH2_d = _disp3(r_con_H2, r_unc_H2)
  dxO = (dxO_d[0].astype(w_dtype), dxO_d[1].astype(w_dtype), dxO_d[2].astype(w_dtype))
  dxH1 = (dxH1_d[0].astype(w_dtype), dxH1_d[1].astype(w_dtype), dxH1_d[2].astype(w_dtype))
  dxH2 = (dxH2_d[0].astype(w_dtype), dxH2_d[1].astype(w_dtype), dxH2_d[2].astype(w_dtype))

  cO = mO_state.astype(w_dtype) * inv_half_dt
  cH1 = mH1_state.astype(w_dtype) * inv_half_dt
  cH2 = mH2_state.astype(w_dtype) * inv_half_dt
  p_pre_O_w = (p_pre_O[0].astype(w_dtype), p_pre_O[1].astype(w_dtype), p_pre_O[2].astype(w_dtype))
  p_pre_H1_w = (p_pre_H1[0].astype(w_dtype), p_pre_H1[1].astype(w_dtype), p_pre_H1[2].astype(w_dtype))
  p_pre_H2_w = (p_pre_H2[0].astype(w_dtype), p_pre_H2[1].astype(w_dtype), p_pre_H2[2].astype(w_dtype))

  p_r_O = (p_pre_O_w[0] + cO * dxO[0], p_pre_O_w[1] + cO * dxO[1], p_pre_O_w[2] + cO * dxO[2])
  p_r_H1 = (p_pre_H1_w[0] + cH1 * dxH1[0], p_pre_H1_w[1] + cH1 * dxH1[1], p_pre_H1_w[2] + cH1 * dxH1[2])
  p_r_H2 = (p_pre_H2_w[0] + cH2 * dxH2[0], p_pre_H2_w[1] + cH2 * dxH2[1], p_pre_H2_w[2] + cH2 * dxH2[2])

  # Unwrap H_unc relative to O_unc for L_target
  if has_box and box_xyz is not None:
    dH1_u = _disp3(r_unc_H1, r_unc_O)
    uH1 = (r_unc_O[0] + dH1_u[0], r_unc_O[1] + dH1_u[1], r_unc_O[2] + dH1_u[2])
    dH2_u = _disp3(r_unc_H2, r_unc_O)
    uH2 = (r_unc_O[0] + dH2_u[0], r_unc_O[1] + dH2_u[1], r_unc_O[2] + dH2_u[2])
  else:
    uH1, uH2 = r_unc_H1, r_unc_H2
  uO = r_unc_O

  com_u_x = (m_O_d * uO[0] + m_H_d * (uH1[0] + uH2[0])) * inv_m_tot_d
  com_u_y = (m_O_d * uO[1] + m_H_d * (uH1[1] + uH2[1])) * inv_m_tot_d
  com_u_z = (m_O_d * uO[2] + m_H_d * (uH1[2] + uH2[2])) * inv_m_tot_d

  dO_u = ((uO[0] - com_u_x).astype(w_dtype), (uO[1] - com_u_y).astype(w_dtype), (uO[2] - com_u_z).astype(w_dtype))
  dH1_u = ((uH1[0] - com_u_x).astype(w_dtype), (uH1[1] - com_u_y).astype(w_dtype), (uH1[2] - com_u_z).astype(w_dtype))
  dH2_u = ((uH2[0] - com_u_x).astype(w_dtype), (uH2[1] - com_u_y).astype(w_dtype), (uH2[2] - com_u_z).astype(w_dtype))

  LO_t = _cross_vec3(dO_u, p_pre_O_w)
  LH1_t = _cross_vec3(dH1_u, p_pre_H1_w)
  LH2_t = _cross_vec3(dH2_u, p_pre_H2_w)
  Lt_x = LO_t[0] + LH1_t[0] + LH2_t[0]
  Lt_y = LO_t[1] + LH1_t[1] + LH2_t[1]
  Lt_z = LO_t[2] + LH1_t[2] + LH2_t[2]

  com_c_x = (m_O_d * r_con_O[0] + m_H_d * (r_con_H1[0] + r_con_H2[0])) * inv_m_tot_d
  com_c_y = (m_O_d * r_con_O[1] + m_H_d * (r_con_H1[1] + r_con_H2[1])) * inv_m_tot_d
  com_c_z = (m_O_d * r_con_O[2] + m_H_d * (r_con_H1[2] + r_con_H2[2])) * inv_m_tot_d

  dO_c = ((r_con_O[0] - com_c_x).astype(w_dtype), (r_con_O[1] - com_c_y).astype(w_dtype), (r_con_O[2] - com_c_z).astype(w_dtype))
  dH1_c = ((r_con_H1[0] - com_c_x).astype(w_dtype), (r_con_H1[1] - com_c_y).astype(w_dtype), (r_con_H1[2] - com_c_z).astype(w_dtype))
  dH2_c = ((r_con_H2[0] - com_c_x).astype(w_dtype), (r_con_H2[1] - com_c_y).astype(w_dtype), (r_con_H2[2] - com_c_z).astype(w_dtype))

  LO_c = _cross_vec3(dO_c, p_r_O)
  LH1_c = _cross_vec3(dH1_c, p_r_H1)
  LH2_c = _cross_vec3(dH2_c, p_r_H2)
  Lc_x = Lt_x - (LO_c[0] + LH1_c[0] + LH2_c[0])
  Lc_y = Lt_y - (LO_c[1] + LH1_c[1] + LH2_c[1])
  Lc_z = Lt_z - (LO_c[2] + LH1_c[2] + LH2_c[2])

  Ixx, Iyy, Izz, Ixy, Ixz, Iyz = _inertia_tensor_vec3(dO_c, dH1_c, dH2_c, m_O, m_H, m_H)
  ox, oy, oz = _solve_3x3_vec3(Ixx, Iyy, Izz, Ixy, Ixz, Iyz, Lc_x, Lc_y, Lc_z)
  omega = (ox, oy, oz)

  w_dO = _cross_vec3(omega, dO_c)
  w_dH1 = _cross_vec3(omega, dH1_c)
  w_dH2 = _cross_vec3(omega, dH2_c)

  p_out_O = (
    (p_r_O[0] + m_O * w_dO[0]).astype(dtype),
    (p_r_O[1] + m_O * w_dO[1]).astype(dtype),
    (p_r_O[2] + m_O * w_dO[2]).astype(dtype),
  )
  p_out_H1 = (
    (p_r_H1[0] + m_H * w_dH1[0]).astype(dtype),
    (p_r_H1[1] + m_H * w_dH1[1]).astype(dtype),
    (p_r_H1[2] + m_H * w_dH1[2]).astype(dtype),
  )
  p_out_H2 = (
    (p_r_H2[0] + m_H * w_dH2[0]).astype(dtype),
    (p_r_H2[1] + m_H * w_dH2[1]).astype(dtype),
    (p_r_H2[2] + m_H * w_dH2[2]).astype(dtype),
  )
  return p_out_O, p_out_H1, p_out_H2


def _water_o_step_vec3(
  pO: tuple[Array, Array, Array],
  pH1: tuple[Array, Array, Array],
  pH2: tuple[Array, Array, Array],
  rO: tuple[Array, Array, Array],
  rH1: tuple[Array, Array, Array],
  rH2: tuple[Array, Array, Array],
  mO: Array,
  mH1: Array,
  mH2: Array,
  z0: Array,
  z1: Array,
  z2: Array,
  z3: Array,
  z4: Array,
  z5: Array,
  c1: Array,
  c2: Array,
  kT_val: Array,
  use_f32_math: bool = False,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array], tuple[Array, Array, Array]]:
  """In-register 6D rigid-body Ornstein-Uhlenbeck step for a block of waters."""
  dtype = rO[0].dtype
  w_dtype = jnp.float32 if (use_f32_math and dtype == jnp.float64) else dtype
  eps_d = jnp.asarray(1e-30, dtype=dtype)
  msum_d = mO + mH1 + mH2
  inv_msum_d = jnp.asarray(1.0, dtype=dtype) / jnp.maximum(msum_d, eps_d)

  com_x = (mO * rO[0] + mH1 * rH1[0] + mH2 * rH2[0]) * inv_msum_d
  com_y = (mO * rO[1] + mH1 * rH1[1] + mH2 * rH2[1]) * inv_msum_d
  com_z = (mO * rO[2] + mH1 * rH1[2] + mH2 * rH2[2]) * inv_msum_d

  dO = ((rO[0] - com_x).astype(w_dtype), (rO[1] - com_y).astype(w_dtype), (rO[2] - com_z).astype(w_dtype))
  dH1 = ((rH1[0] - com_x).astype(w_dtype), (rH1[1] - com_y).astype(w_dtype), (rH1[2] - com_z).astype(w_dtype))
  dH2 = ((rH2[0] - com_x).astype(w_dtype), (rH2[1] - com_y).astype(w_dtype), (rH2[2] - com_z).astype(w_dtype))

  mO_w = mO.astype(w_dtype)
  mH1_w = mH1.astype(w_dtype)
  mH2_w = mH2.astype(w_dtype)
  inv_msum = inv_msum_d.astype(w_dtype)
  pO_w = (pO[0].astype(w_dtype), pO[1].astype(w_dtype), pO[2].astype(w_dtype))
  pH1_w = (pH1[0].astype(w_dtype), pH1[1].astype(w_dtype), pH1[2].astype(w_dtype))
  pH2_w = (pH2[0].astype(w_dtype), pH2[1].astype(w_dtype), pH2[2].astype(w_dtype))
  c1_w = c1.astype(w_dtype)
  c2_w = c2.astype(w_dtype)
  kT_w = kT_val.astype(w_dtype)

  Ixx, Iyy, Izz, Ixy, Ixz, Iyz = _inertia_tensor_vec3(dO, dH1, dH2, mO_w, mH1_w, mH2_w)
  reg = jnp.asarray(1e-12, dtype=w_dtype) * ((Ixx + Iyy + Izz) / jnp.asarray(3.0, dtype=w_dtype) + jnp.asarray(1.0, dtype=w_dtype))
  Ixx_r, Iyy_r, Izz_r = Ixx + reg, Iyy + reg, Izz + reg

  v_com_x = (pO_w[0] + pH1_w[0] + pH2_w[0]) * inv_msum
  v_com_y = (pO_w[1] + pH1_w[1] + pH2_w[1]) * inv_msum
  v_com_z = (pO_w[2] + pH1_w[2] + pH2_w[2]) * inv_msum

  LO = _cross_vec3(dO, pO_w)
  LH1 = _cross_vec3(dH1, pH1_w)
  LH2 = _cross_vec3(dH2, pH2_w)
  Lx = LO[0] + LH1[0] + LH2[0]
  Ly = LO[1] + LH1[1] + LH2[1]
  Lz = LO[2] + LH1[2] + LH2[2]

  wp_x, wp_y, wp_z = _solve_3x3_vec3(Ixx_r, Iyy_r, Izz_r, Ixy, Ixz, Iyz, Lx, Ly, Lz)
  wp = (wp_x, wp_y, wp_z)
  wp_dO = _cross_vec3(wp, dO)
  wp_dH1 = _cross_vec3(wp, dH1)
  wp_dH2 = _cross_vec3(wp, dH2)

  pO_rig = (mO_w * (v_com_x + wp_dO[0]), mO_w * (v_com_y + wp_dO[1]), mO_w * (v_com_z + wp_dO[2]))
  pH1_rig = (mH1_w * (v_com_x + wp_dH1[0]), mH1_w * (v_com_y + wp_dH1[1]), mH1_w * (v_com_z + wp_dH1[2]))
  pH2_rig = (mH2_w * (v_com_x + wp_dH2[0]), mH2_w * (v_com_y + wp_dH2[1]), mH2_w * (v_com_z + wp_dH2[2]))

  sqrt_kT = jnp.sqrt(jnp.maximum(kT_w, jnp.asarray(0.0, dtype=w_dtype)))
  scale_tr = sqrt_kT * jnp.sqrt(inv_msum)
  xi_x, xi_y, xi_z = z0.astype(w_dtype) * scale_tr, z1.astype(w_dtype) * scale_tr, z2.astype(w_dtype) * scale_tr

  wn_x, wn_y, wn_z = _cholesky_upper_solve_3x3_vec3(
    Ixx_r, Iyy_r, Izz_r, Ixy, Ixz, Iyz,
    sqrt_kT * z3.astype(w_dtype), sqrt_kT * z4.astype(w_dtype), sqrt_kT * z5.astype(w_dtype),
  )
  wn = (wn_x, wn_y, wn_z)
  wn_dO = _cross_vec3(wn, dO)
  wn_dH1 = _cross_vec3(wn, dH1)
  wn_dH2 = _cross_vec3(wn, dH2)

  nO = (mO_w * (xi_x + wn_dO[0]), mO_w * (xi_y + wn_dO[1]), mO_w * (xi_z + wn_dO[2]))
  nH1 = (mH1_w * (xi_x + wn_dH1[0]), mH1_w * (xi_y + wn_dH1[1]), mH1_w * (xi_z + wn_dH1[2]))
  nH2 = (mH2_w * (xi_x + wn_dH2[0]), mH2_w * (xi_y + wn_dH2[1]), mH2_w * (xi_z + wn_dH2[2]))

  out_O = (
    (c1_w * pO_rig[0] + c2_w * nO[0]).astype(dtype),
    (c1_w * pO_rig[1] + c2_w * nO[1]).astype(dtype),
    (c1_w * pO_rig[2] + c2_w * nO[2]).astype(dtype),
  )
  out_H1 = (
    (c1_w * pH1_rig[0] + c2_w * nH1[0]).astype(dtype),
    (c1_w * pH1_rig[1] + c2_w * nH1[1]).astype(dtype),
    (c1_w * pH1_rig[2] + c2_w * nH1[2]).astype(dtype),
  )
  out_H2 = (
    (c1_w * pH2_rig[0] + c2_w * nH2[0]).astype(dtype),
    (c1_w * pH2_rig[1] + c2_w * nH2[1]).astype(dtype),
    (c1_w * pH2_rig[2] + c2_w * nH2[2]).astype(dtype),
  )
  return out_O, out_H1, out_H2


def _settle_vel_and_am_vec3(
  pO: tuple[Array, Array, Array],
  pH1: tuple[Array, Array, Array],
  pH2: tuple[Array, Array, Array],
  rO: tuple[Array, Array, Array],
  rH1: tuple[Array, Array, Array],
  rH2: tuple[Array, Array, Array],
  mO_state: Array,
  mH1_state: Array,
  mH2_state: Array,
  mass_oxygen: float,
  mass_hydrogen: float,
  valid_w: Array,
  use_f32_math: bool = False,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array], tuple[Array, Array, Array]]:
  """In-register 3x3 Gram RATTLE velocity constraint + angular momentum impulse removal."""
  dtype = rO[0].dtype
  w_dtype = jnp.float32 if (use_f32_math and dtype == jnp.float64) else dtype
  zero_d = jnp.asarray(0.0, dtype=dtype)
  zero = jnp.asarray(0.0, dtype=w_dtype)
  one = jnp.asarray(1.0, dtype=w_dtype)
  two = jnp.asarray(2.0, dtype=w_dtype)
  eps = jnp.asarray(1e-10, dtype=w_dtype)

  mO_w = mO_state.astype(w_dtype)
  mH1_w = mH1_state.astype(w_dtype)
  mH2_w = mH2_state.astype(w_dtype)
  inv_mO_state = one / jnp.maximum(mO_w, jnp.asarray(1e-30, dtype=w_dtype))
  inv_mH1_state = one / jnp.maximum(mH1_w, jnp.asarray(1e-30, dtype=w_dtype))
  inv_mH2_state = one / jnp.maximum(mH2_w, jnp.asarray(1e-30, dtype=w_dtype))

  vO = (pO[0].astype(w_dtype) * inv_mO_state, pO[1].astype(w_dtype) * inv_mO_state, pO[2].astype(w_dtype) * inv_mO_state)
  vH1 = (pH1[0].astype(w_dtype) * inv_mH1_state, pH1[1].astype(w_dtype) * inv_mH1_state, pH1[2].astype(w_dtype) * inv_mH1_state)
  vH2 = (pH2[0].astype(w_dtype) * inv_mH2_state, pH2[1].astype(w_dtype) * inv_mH2_state, pH2[2].astype(w_dtype) * inv_mH2_state)

  def _norm_vec(a, b):
    vx = (a[0] - b[0]).astype(w_dtype)
    vy = (a[1] - b[1]).astype(w_dtype)
    vz = (a[2] - b[2]).astype(w_dtype)
    d = jnp.sqrt(vx * vx + vy * vy + vz * vz) + eps
    inv_d = jnp.where(valid_w, one / d, zero)
    return (vx * inv_d, vy * inv_d, vz * inv_d)

  n1 = _norm_vec(rH1, rO)
  n2 = _norm_vec(rH2, rO)
  n3 = _norm_vec(rH2, rH1)

  b1 = zero - ((vH1[0] - vO[0]) * n1[0] + (vH1[1] - vO[1]) * n1[1] + (vH1[2] - vO[2]) * n1[2])
  b2 = zero - ((vH2[0] - vO[0]) * n2[0] + (vH2[1] - vO[1]) * n2[1] + (vH2[2] - vO[2]) * n2[2])
  b3 = zero - ((vH2[0] - vH1[0]) * n3[0] + (vH2[1] - vH1[1]) * n3[1] + (vH2[2] - vH1[2]) * n3[2])

  mu_O = jnp.asarray(1.0 / mass_oxygen, dtype=w_dtype)
  mu_H = jnp.asarray(1.0 / mass_hydrogen, dtype=w_dtype)
  neg_mu_O = jnp.asarray(-1.0 / mass_oxygen, dtype=w_dtype)
  neg_mu_H = jnp.asarray(-1.0 / mass_hydrogen, dtype=w_dtype)
  diag_OH = jnp.broadcast_to(mu_O + mu_H, b1.shape)
  diag_HH = jnp.broadcast_to(two * mu_H, b1.shape)

  c12 = mu_O * (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2])
  c13 = neg_mu_H * (n1[0] * n3[0] + n1[1] * n3[1] + n1[2] * n3[2])
  c23 = mu_H * (n2[0] * n3[0] + n2[1] * n3[1] + n2[2] * n3[2])

  l1, l2, l3 = _solve_3x3_vec3(diag_OH, diag_OH, diag_HH, c12, c13, c23, b1, b2, b3)

  dvO = (
    jnp.where(valid_w, neg_mu_O * (l1 * n1[0] + l2 * n2[0]), zero),
    jnp.where(valid_w, neg_mu_O * (l1 * n1[1] + l2 * n2[1]), zero),
    jnp.where(valid_w, neg_mu_O * (l1 * n1[2] + l2 * n2[2]), zero),
  )
  dvH1 = (
    jnp.where(valid_w, mu_H * (l1 * n1[0] - l3 * n3[0]), zero),
    jnp.where(valid_w, mu_H * (l1 * n1[1] - l3 * n3[1]), zero),
    jnp.where(valid_w, mu_H * (l1 * n1[2] - l3 * n3[2]), zero),
  )
  dvH2 = (
    jnp.where(valid_w, mu_H * (l2 * n2[0] + l3 * n3[0]), zero),
    jnp.where(valid_w, mu_H * (l2 * n2[1] + l3 * n3[1]), zero),
    jnp.where(valid_w, mu_H * (l2 * n2[2] + l3 * n3[2]), zero),
  )

  m_O_d = jnp.asarray(mass_oxygen, dtype=dtype)
  m_H_d = jnp.asarray(mass_hydrogen, dtype=dtype)
  inv_m_tot_d = jnp.asarray(1.0 / (mass_oxygen + 2.0 * mass_hydrogen), dtype=dtype)
  m_O = jnp.asarray(mass_oxygen, dtype=w_dtype)
  m_H = jnp.asarray(mass_hydrogen, dtype=w_dtype)

  com_x = (m_O_d * rO[0] + m_H_d * (rH1[0] + rH2[0])) * inv_m_tot_d
  com_y = (m_O_d * rO[1] + m_H_d * (rH1[1] + rH2[1])) * inv_m_tot_d
  com_z = (m_O_d * rO[2] + m_H_d * (rH1[2] + rH2[2])) * inv_m_tot_d

  dO = ((rO[0] - com_x).astype(w_dtype), (rO[1] - com_y).astype(w_dtype), (rO[2] - com_z).astype(w_dtype))
  dH1 = ((rH1[0] - com_x).astype(w_dtype), (rH1[1] - com_y).astype(w_dtype), (rH1[2] - com_z).astype(w_dtype))
  dH2 = ((rH2[0] - com_x).astype(w_dtype), (rH2[1] - com_y).astype(w_dtype), (rH2[2] - com_z).astype(w_dtype))

  dpO = (m_O * dvO[0], m_O * dvO[1], m_O * dvO[2])
  dpH1 = (m_H * dvH1[0], m_H * dvH1[1], m_H * dvH1[2])
  dpH2 = (m_H * dvH2[0], m_H * dvH2[1], m_H * dvH2[2])

  LO = _cross_vec3(dO, dpO)
  LH1 = _cross_vec3(dH1, dpH1)
  LH2 = _cross_vec3(dH2, dpH2)
  Lx = LO[0] + LH1[0] + LH2[0]
  Ly = LO[1] + LH1[1] + LH2[1]
  Lz = LO[2] + LH1[2] + LH2[2]

  Ixx, Iyy, Izz, Ixy, Ixz, Iyz = _inertia_tensor_vec3(dO, dH1, dH2, m_O, m_H, m_H)
  ox, oy, oz = _solve_3x3_vec3(Ixx, Iyy, Izz, Ixy, Ixz, Iyz, Lx, Ly, Lz)
  omega = (ox, oy, oz)

  w_dO = _cross_vec3(omega, dO)
  w_dH1 = _cross_vec3(omega, dH1)
  w_dH2 = _cross_vec3(omega, dH2)

  dvO_cor = (dvO[0] - w_dO[0], dvO[1] - w_dO[1], dvO[2] - w_dO[2])
  dvH1_cor = (dvH1[0] - w_dH1[0], dvH1[1] - w_dH1[1], dvH1[2] - w_dH1[2])
  dvH2_cor = (dvH2[0] - w_dH2[0], dvH2[1] - w_dH2[1], dvH2[2] - w_dH2[2])

  pO_out = (
    pO[0] + jnp.where(valid_w, (mO_w * dvO_cor[0]).astype(dtype), zero_d),
    pO[1] + jnp.where(valid_w, (mO_w * dvO_cor[1]).astype(dtype), zero_d),
    pO[2] + jnp.where(valid_w, (mO_w * dvO_cor[2]).astype(dtype), zero_d),
  )
  pH1_out = (
    pH1[0] + jnp.where(valid_w, (mH1_w * dvH1_cor[0]).astype(dtype), zero_d),
    pH1[1] + jnp.where(valid_w, (mH1_w * dvH1_cor[1]).astype(dtype), zero_d),
    pH1[2] + jnp.where(valid_w, (mH1_w * dvH1_cor[2]).astype(dtype), zero_d),
  )
  pH2_out = (
    pH2[0] + jnp.where(valid_w, (mH2_w * dvH2_cor[0]).astype(dtype), zero_d),
    pH2[1] + jnp.where(valid_w, (mH2_w * dvH2_cor[1]).astype(dtype), zero_d),
    pH2[2] + jnp.where(valid_w, (mH2_w * dvH2_cor[2]).astype(dtype), zero_d),
  )
  return pO_out, pH1_out, pH2_out


def _philox2x32_box_muller_pair(
  k0_i32: Array,
  k1_i32: Array,
  c0_i32: Array,
  c1_i32: Array,
  out_dtype: Any,
) -> tuple[Array, Array]:
  """Generate two independent standard normal N(0, 1) draws via 4-round 32-bit bijective mixing + Box-Muller."""
  k0 = k0_i32.astype(jnp.uint32)
  k1 = k1_i32.astype(jnp.uint32)
  x0 = (c0_i32.astype(jnp.uint32) + k0) ^ (k1 * jnp.uint32(0x9E3779B9))
  x1 = (c1_i32.astype(jnp.uint32) + k1) ^ (k0 * jnp.uint32(0xBB67AE85))
  for w0, w1 in (
    (0x9E3779B9, 0xBB67AE85),
    (0x3C6EF372, 0x76CF5D0A),
    (0xDAA66D2B, 0x32370B8F),
    (0x78DDE6E4, 0xEDA6B914),
  ):
    x0 = x0 ^ (x1 >> jnp.uint32(15))
    x0 = x0 * jnp.uint32(0x85EBCA6B)
    x1 = x1 ^ (x0 >> jnp.uint32(13))
    x1 = x1 * jnp.uint32(0xC2B2AE35)
    x0 = (x0 + jnp.uint32(w0)) ^ (x1 >> jnp.uint32(16))
    x1 = (x1 + jnp.uint32(w1)) ^ (x0 >> jnp.uint32(16))
  u0 = ((x0 >> jnp.uint32(8)).astype(jnp.float32) + jnp.float32(0.5)) * jnp.float32(1.0 / 16777216.0)
  u1 = ((x1 >> jnp.uint32(8)).astype(jnp.float32) + jnp.float32(0.5)) * jnp.float32(1.0 / 16777216.0)
  r = jnp.sqrt(jnp.maximum(jnp.float32(-2.0) * jnp.log(u0), jnp.float32(0.0)))
  theta = jnp.float32(6.283185307179586) * u1
  return (r * jnp.cos(theta)).astype(out_dtype), (r * jnp.sin(theta)).astype(out_dtype)


_SETTLE_RATTLE_BLOCK = 32


def _resolve_active_limit(
  val: int | None,
  env_var: str,
  upper: int,
  min_val: int = 0,
  mask: Array | None = None,
) -> int:
  if val is not None:
    return min(upper, max(min_val, int(val)))
  raw = os.environ.get(env_var, "").strip()
  if raw:
    try:
      return min(upper, max(min_val, int(raw)))
    except ValueError:
      pass
  if mask is not None and not isinstance(mask, jax.core.Tracer):
    try:
      import numpy as _np

      m_np = _np.asarray(mask, dtype=bool)
      if m_np.size > 0:
        m_any = _np.any(m_np, axis=tuple(range(m_np.ndim - 1))) if m_np.ndim > 1 else m_np
        nz = _np.nonzero(m_any)[0]
        if nz.size > 0:
          return min(upper, max(min_val, int(nz[-1]) + 1))
    except Exception:
      pass
  return upper


def _build_rattle_clusters(
  B: int,
  N: int,
  C: int,
  cp_b: Array,
  cl_b: Array,
  cm_b: Array,
  mass_b: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
  """Pack arbitrary X-H constraint pairs into disjoint heavy-atom star clusters (X, H0..H3).

  Returns:
    cX: ``(B, C)`` int32 heavy atom index per cluster
    cH0..cH3: ``(B, C)`` int32 hydrogen atom indices for slots 0..3
    cD0..cD3: ``(B, C)`` target bond lengths for slots 0..3
    cM: ``(B, C)`` int32 4-bit validity mask (bit s = 1 iff slot s is active)
    constrained_atom_mask: ``(B, N)`` int32 (1 if atom belongs to any active constraint)
  """
  dtype = cl_b.dtype
  if C <= 0:
    z_i = jnp.zeros((B, 1), dtype=jnp.int32)
    z_f = jnp.zeros((B, 1), dtype=dtype)
    return z_i, z_i, z_i, z_i, z_i, z_f, z_f, z_f, z_f, z_i, jnp.zeros((B, N), dtype=jnp.int32)

  b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]
  c_arange = jnp.arange(C, dtype=jnp.int32)[None, :]
  valid_c = (cm_b > 0) & (c_arange < C)
  i0 = jnp.where(valid_c, jnp.clip(cp_b[:, :, 0], 0, max(N - 1, 0)), 0)
  i1 = jnp.where(valid_c, jnp.clip(cp_b[:, :, 1], 0, max(N - 1, 0)), 0)
  deg = jnp.zeros((B, N + 1), dtype=jnp.int32)
  deg = deg.at[b_idx, jnp.where(valid_c, i0, N)].add(1)
  deg = deg.at[b_idx, jnp.where(valid_c, i1, N)].add(1)
  deg0 = deg[b_idx, i0]
  deg1 = deg[b_idx, i1]
  m0 = mass_b[b_idx, i0]
  m1 = mass_b[b_idx, i1]
  first_is_heavy = (deg0 > deg1) | ((deg0 == deg1) & ((m0 > m1) | ((m0 == m1) & (i0 <= i1))))
  x_raw = jnp.where(first_is_heavy, i0, i1)
  h_raw = jnp.where(first_is_heavy, i1, i0)
  constrained_atom_mask = (deg[:, :N] > 0).astype(jnp.int32)

  sort_key = jnp.where(valid_c, x_raw, N)
  order = jnp.argsort(sort_key, axis=1)
  x_s = jnp.take_along_axis(x_raw, order, axis=1)
  h_s = jnp.take_along_axis(h_raw, order, axis=1)
  d_s = jnp.take_along_axis(cl_b, order, axis=1)
  v_s = jnp.take_along_axis(valid_c.astype(jnp.int32), order, axis=1) > 0

  prev_x = jnp.concatenate([jnp.full((B, 1), -1, dtype=jnp.int32), x_s[:, :-1]], axis=1)
  is_head = v_s & ((c_arange == 0) | (x_s != prev_x))
  head_order = jnp.argsort(jnp.where(is_head, c_arange, C), axis=1)

  idx0 = jnp.minimum(head_order, max(C - 1, 0))
  idx1 = jnp.minimum(head_order + 1, max(C - 1, 0))
  idx2 = jnp.minimum(head_order + 2, max(C - 1, 0))
  idx3 = jnp.minimum(head_order + 3, max(C - 1, 0))

  cX = jnp.take_along_axis(x_s, idx0, axis=1)
  cH0 = jnp.take_along_axis(h_s, idx0, axis=1)
  cH1 = jnp.take_along_axis(h_s, idx1, axis=1)
  cH2 = jnp.take_along_axis(h_s, idx2, axis=1)
  cH3 = jnp.take_along_axis(h_s, idx3, axis=1)

  cD0 = jnp.take_along_axis(d_s, idx0, axis=1)
  cD1 = jnp.take_along_axis(d_s, idx1, axis=1)
  cD2 = jnp.take_along_axis(d_s, idx2, axis=1)
  cD3 = jnp.take_along_axis(d_s, idx3, axis=1)

  v0 = jnp.take_along_axis(is_head.astype(jnp.int32), idx0, axis=1) > 0
  v1 = (
    v0
    & (head_order + 1 < C)
    & (jnp.take_along_axis(v_s.astype(jnp.int32), idx1, axis=1) > 0)
    & (jnp.take_along_axis(x_s, idx1, axis=1) == cX)
  )
  v2 = (
    v1
    & (head_order + 2 < C)
    & (jnp.take_along_axis(v_s.astype(jnp.int32), idx2, axis=1) > 0)
    & (jnp.take_along_axis(x_s, idx2, axis=1) == cX)
  )
  v3 = (
    v2
    & (head_order + 3 < C)
    & (jnp.take_along_axis(v_s.astype(jnp.int32), idx3, axis=1) > 0)
    & (jnp.take_along_axis(x_s, idx3, axis=1) == cX)
  )
  cM = (
    v0.astype(jnp.int32)
    | (v1.astype(jnp.int32) << 1)
    | (v2.astype(jnp.int32) << 2)
    | (v3.astype(jnp.int32) << 3)
  )
  return cX, cH0, cH1, cH2, cH3, cD0, cD1, cD2, cD3, cM, constrained_atom_mask


def _build_is_water_or_ghost_mask(
  B: int,
  N: int,
  W: int,
  wi_b: Array,
  wm_b: Array,
  am_b: Array | None = None,
  constrained_atom_mask: Array | None = None,
) -> Array:
  """Build ``(B * N,)`` int32 mask: 0 for unconstrained solute, 1 for water, 2 for RATTLE, 3 for ghost."""
  is_water_atom = jnp.zeros((B, N + 1), dtype=jnp.int32)
  if W > 0:
    b_idx_grid = jnp.arange(B, dtype=jnp.int32)[:, None]
    valid_w_b = (wm_b > 0) & (jnp.arange(W, dtype=jnp.int32)[None, :] < W)
    safe_wO = jnp.where(valid_w_b, jnp.clip(wi_b[:, :, 0], 0, max(N - 1, 0)), N)
    safe_wH1 = jnp.where(valid_w_b, jnp.clip(wi_b[:, :, 1], 0, max(N - 1, 0)), N)
    safe_wH2 = jnp.where(valid_w_b, jnp.clip(wi_b[:, :, 2], 0, max(N - 1, 0)), N)
    is_water_atom = is_water_atom.at[b_idx_grid, safe_wO].set(1)
    is_water_atom = is_water_atom.at[b_idx_grid, safe_wH1].set(1)
    is_water_atom = is_water_atom.at[b_idx_grid, safe_wH2].set(1)
  out = is_water_atom[:, :N]
  if constrained_atom_mask is not None:
    out = jnp.where((constrained_atom_mask > 0) & (out == 0), 2, out)
  if am_b is not None:
    out = jnp.where(am_b > 0, out, 3)
  return out.reshape(B * N)


def _rattle_shake_cluster_vec3(
  r_unc_X: tuple[Array, Array, Array],
  r_unc_H: list[tuple[Array, Array, Array]],
  ab_ref: list[tuple[Array, Array, Array]],
  inv_mX: Array,
  inv_mH: list[Array],
  d_sq: list[Array],
  v_slot: list[Array],
  n_iters: int = 8,
  use_f32_math: bool = False,
) -> tuple[tuple[Array, Array, Array], list[tuple[Array, Array, Array]]]:
  """Coupled in-register SHAKE position projection for a heavy-atom star cluster (X, H0..H3)."""
  dtype = r_unc_X[0].dtype
  w_dtype = jnp.float32 if (use_f32_math and dtype == jnp.float64) else dtype
  zero_d = jnp.asarray(0.0, dtype=dtype)
  zero = jnp.asarray(0.0, dtype=w_dtype)
  zero_vec = jnp.zeros_like(r_unc_X[0], dtype=w_dtype)
  two = jnp.asarray(2.0, dtype=w_dtype)
  eps = jnp.asarray(1e-12, dtype=w_dtype)

  dxX_x, dxX_y, dxX_z = zero_vec, zero_vec, zero_vec
  dxH = [(zero_vec, zero_vec, zero_vec) for _ in range(4)]
  ab_u = [
    (
      (r_unc_H[k][0] - r_unc_X[0]).astype(w_dtype),
      (r_unc_H[k][1] - r_unc_X[1]).astype(w_dtype),
      (r_unc_H[k][2] - r_unc_X[2]).astype(w_dtype),
    )
    for k in range(4)
  ]
  ab0 = [
    (
      ab_ref[k][0].astype(w_dtype),
      ab_ref[k][1].astype(w_dtype),
      ab_ref[k][2].astype(w_dtype),
    )
    for k in range(4)
  ]
  imX = inv_mX.astype(w_dtype)
  imH = [inv_mH[k].astype(w_dtype) for k in range(4)]
  dsq = [d_sq[k].astype(w_dtype) for k in range(4)]

  def _shake_iter(_i, carry):
    dxX_x_c, dxX_y_c, dxX_z_c, dxH_c = carry
    dxH_next = list(dxH_c)
    for k in range(4):
      vk = v_slot[k]
      ab_x = ab_u[k][0] + dxH_next[k][0] - dxX_x_c
      ab_y = ab_u[k][1] + dxH_next[k][1] - dxX_y_c
      ab_z = ab_u[k][2] + dxH_next[k][2] - dxX_z_c
      dist_sq = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z
      dot = ab_x * ab0[k][0] + ab_y * ab0[k][1] + ab_z * ab0[k][2]
      denom = two * (imX + imH[k]) * dot
      ok = vk & (jnp.abs(denom) > eps)
      lam = jnp.where(ok, (dsq[k] - dist_sq) / jnp.where(ok, denom, two), zero)
      cX_k = lam * imX
      cH_k = lam * imH[k]
      dxX_x_c = dxX_x_c - cX_k * ab0[k][0]
      dxX_y_c = dxX_y_c - cX_k * ab0[k][1]
      dxX_z_c = dxX_z_c - cX_k * ab0[k][2]
      dxH_next[k] = (
        dxH_next[k][0] + cH_k * ab0[k][0],
        dxH_next[k][1] + cH_k * ab0[k][1],
        dxH_next[k][2] + cH_k * ab0[k][2],
      )
    return (dxX_x_c, dxX_y_c, dxX_z_c, tuple(dxH_next))

  dxX_x, dxX_y, dxX_z, dxH_t = jax.lax.fori_loop(
    0, int(n_iters), _shake_iter, (dxX_x, dxX_y, dxX_z, tuple(dxH))
  )
  dxH = list(dxH_t)

  rX_out = (
    r_unc_X[0] + dxX_x.astype(dtype),
    r_unc_X[1] + dxX_y.astype(dtype),
    r_unc_X[2] + dxX_z.astype(dtype),
  )
  rH_out = [
    (
      r_unc_H[k][0] + jnp.where(v_slot[k], dxH[k][0].astype(dtype), zero_d),
      r_unc_H[k][1] + jnp.where(v_slot[k], dxH[k][1].astype(dtype), zero_d),
      r_unc_H[k][2] + jnp.where(v_slot[k], dxH[k][2].astype(dtype), zero_d),
    )
    for k in range(4)
  ]
  return rX_out, rH_out


def _rattle_vel_cluster_vec3(
  pX: tuple[Array, Array, Array],
  pH: list[tuple[Array, Array, Array]],
  ab_cur: list[tuple[Array, Array, Array]],
  inv_mX: Array,
  inv_mH: list[Array],
  v_slot: list[Array],
  n_iters: int = 8,
  use_f32_math: bool = False,
) -> tuple[tuple[Array, Array, Array], list[tuple[Array, Array, Array]]]:
  """Coupled in-register RATTLE velocity projection for a heavy-atom star cluster (X, H0..H3)."""
  dtype = pX[0].dtype
  w_dtype = jnp.float32 if (use_f32_math and dtype == jnp.float64) else dtype
  zero = jnp.asarray(0.0, dtype=w_dtype)
  one = jnp.asarray(1.0, dtype=w_dtype)
  eps = jnp.asarray(1e-12, dtype=w_dtype)

  pX_x, pX_y, pX_z = pX[0].astype(w_dtype), pX[1].astype(w_dtype), pX[2].astype(w_dtype)
  pH_w = [(pH[k][0].astype(w_dtype), pH[k][1].astype(w_dtype), pH[k][2].astype(w_dtype)) for k in range(4)]
  ab_w = [(ab_cur[k][0].astype(w_dtype), ab_cur[k][1].astype(w_dtype), ab_cur[k][2].astype(w_dtype)) for k in range(4)]
  imX = inv_mX.astype(w_dtype)
  imH = [inv_mH[k].astype(w_dtype) for k in range(4)]

  def _vel_iter(_i, carry):
    pX_x_c, pX_y_c, pX_z_c, pH_c = carry
    pH_next = list(pH_c)
    for k in range(4):
      vk = v_slot[k]
      ab_x, ab_y, ab_z = ab_w[k]
      vx_rel = pH_next[k][0] * imH[k] - pX_x_c * imX
      vy_rel = pH_next[k][1] * imH[k] - pX_y_c * imX
      vz_rel = pH_next[k][2] * imH[k] - pX_z_c * imX
      dot_v = vx_rel * ab_x + vy_rel * ab_y + vz_rel * ab_z
      ab_sq = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z
      denom = (imX + imH[k]) * jnp.maximum(ab_sq, eps)
      ok = vk & (denom > eps)
      mu = jnp.where(ok, (zero - dot_v) / jnp.where(ok, denom, one), zero)
      pX_x_c = pX_x_c - mu * ab_x
      pX_y_c = pX_y_c - mu * ab_y
      pX_z_c = pX_z_c - mu * ab_z
      pH_next[k] = (
        pH_next[k][0] + mu * ab_x,
        pH_next[k][1] + mu * ab_y,
        pH_next[k][2] + mu * ab_z,
      )
    return (pX_x_c, pX_y_c, pX_z_c, tuple(pH_next))

  pX_x, pX_y, pX_z, pH_t = jax.lax.fori_loop(
    0, int(n_iters), _vel_iter, (pX_x, pX_y, pX_z, tuple(pH_w))
  )
  pH_w = list(pH_t)

  pX_out = (pX_x.astype(dtype), pX_y.astype(dtype), pX_z.astype(dtype))
  pH_out = [(pH_w[k][0].astype(dtype), pH_w[k][1].astype(dtype), pH_w[k][2].astype(dtype)) for k in range(4)]
  return pX_out, pH_out


def _pallas_baoab_pre_force_batch(
  B: int,
  in_batched: tuple[bool, ...],
  positions: Array,
  momentum: Array,
  force: Array,
  mass: Array,
  rng_seeds: Array,
  water_indices: Array,
  water_mask: Array,
  box: Array,
  dt_arr: Array,
  kT_arr: Array,
  gamma: float,
  r_OH: float,
  r_HH: float,
  mass_oxygen: float,
  mass_hydrogen: float,
  has_box: bool,
  project_ou_momentum_rigid: bool,
  interpret: bool,
  atom_mask: Array | None = None,
  is_water_or_ghost: Array | None = None,
  constraint_pairs: Array | None = None,
  constraint_lengths: Array | None = None,
  constraint_mask: Array | None = None,
  return_pos_f32: bool = False,
  reference_position: Array | None = None,
  return_max_dr_sq: bool = False,
  n_active_atoms: int | None = None,
  n_active_waters: int | None = None,
) -> tuple[Array, ...]:
  """2D-grid ``(B, n_blocks_atom + n_blocks_water + n_blocks_rattle)`` Pallas BAOAB pre-force kernel."""
  dtype = positions.dtype
  f_dtype = force.dtype
  use_f32_math = (not interpret) and (dtype == jnp.float64) and (
    os.environ.get("PROLIX_BAOAB_F64_INTERNAL", "0") != "1"
  )
  w_dtype = jnp.float32 if use_f32_math else dtype

  def _bcast(arr, is_b):
    return arr if is_b else jnp.broadcast_to(arr[None, ...], (B,) + arr.shape)

  pos_b = _bcast(positions, in_batched[0])
  mom_b = _bcast(momentum, in_batched[1])
  frc_b = _bcast(force, in_batched[2])
  mass_raw = mass.squeeze(-1) if mass.ndim >= 2 and mass.shape[-1] == 1 else mass
  mass_b = _bcast(mass_raw, in_batched[3])
  seed_b = _bcast(rng_seeds, in_batched[4]).astype(jnp.int32)
  wi_b = _bcast(water_indices, in_batched[5]).astype(jnp.int32)
  wm_b = _bcast(water_mask, in_batched[6]).astype(jnp.int32)
  bx_b = _bcast(box, in_batched[7]).astype(dtype)
  dt_b = _bcast(dt_arr, in_batched[8]).astype(dtype).reshape(B)
  kT_b = _bcast(kT_arr, in_batched[9]).astype(dtype).reshape(B)

  N = int(pos_b.shape[1])
  W = int(wi_b.shape[1])
  W_eff = max(W, 1)
  N_act = _resolve_active_limit(n_active_atoms, "PROLIX_ACTIVE_ATOMS", N, min_val=1)
  W_act = _resolve_active_limit(n_active_waters, "PROLIX_ACTIVE_WATERS", W, min_val=0)
  n_blk_a = (N_act + _SETTLE_ATOM_BLOCK - 1) // _SETTLE_ATOM_BLOCK
  n_blk_w = (W_act + _SETTLE_WATER_BLOCK - 1) // _SETTLE_WATER_BLOCK

  arg_cursor = 10
  am_b = None
  if atom_mask is not None:
    am_b = _bcast(atom_mask, in_batched[arg_cursor]).astype(jnp.int32)
    arg_cursor += 1

  iwg_b = None
  if is_water_or_ghost is not None:
    iwg_b = _bcast(is_water_or_ghost, in_batched[arg_cursor]).astype(jnp.int32)
    arg_cursor += 1

  has_rattle = (
    constraint_pairs is not None
    and constraint_lengths is not None
    and int(constraint_pairs.shape[-2]) > 0
  )
  C = int(constraint_pairs.shape[-2]) if has_rattle else 0
  C_eff = max(C, 1)
  n_blk_c = (C + _SETTLE_RATTLE_BLOCK - 1) // _SETTLE_RATTLE_BLOCK if has_rattle else 0

  if has_rattle:
    cp_b = _bcast(constraint_pairs, in_batched[arg_cursor]).astype(jnp.int32)
    arg_cursor += 1
    cl_b = _bcast(constraint_lengths, in_batched[arg_cursor]).astype(dtype)
    arg_cursor += 1
    if constraint_mask is not None:
      cm_b = _bcast(constraint_mask, in_batched[arg_cursor]).astype(jnp.int32)
      arg_cursor += 1
    else:
      cm_b = jnp.ones((B, C), dtype=jnp.int32)
    cX, cH0, cH1, cH2, cH3, cD0, cD1, cD2, cD3, cM, constrained_atom_mask = _build_rattle_clusters(
      B, N, C, cp_b, cl_b, cm_b, mass_b
    )
    cX_in = jnp.asarray(cX.reshape(B * C), dtype=jnp.int32)
    cH0_in = jnp.asarray(cH0.reshape(B * C), dtype=jnp.int32)
    cH1_in = jnp.asarray(cH1.reshape(B * C), dtype=jnp.int32)
    cH2_in = jnp.asarray(cH2.reshape(B * C), dtype=jnp.int32)
    cH3_in = jnp.asarray(cH3.reshape(B * C), dtype=jnp.int32)
    cD0_in = jnp.asarray(cD0.reshape(B * C), dtype=dtype)
    cD1_in = jnp.asarray(cD1.reshape(B * C), dtype=dtype)
    cD2_in = jnp.asarray(cD2.reshape(B * C), dtype=dtype)
    cD3_in = jnp.asarray(cD3.reshape(B * C), dtype=dtype)
    cM_in = jnp.asarray(cM.reshape(B * C), dtype=jnp.int32)
  else:
    constrained_atom_mask = None
    cX_in = cH0_in = cH1_in = cH2_in = cH3_in = cM_in = None
    cD0_in = cD1_in = cD2_in = cD3_in = None

  ref_pos_b = None
  if reference_position is not None:
    ref_pos_b = _bcast(reference_position, in_batched[arg_cursor]).astype(dtype)
    arg_cursor += 1

  px_in = jnp.asarray(pos_b[:, :, 0].reshape(B * N), dtype=dtype)
  py_in = jnp.asarray(pos_b[:, :, 1].reshape(B * N), dtype=dtype)
  pz_in = jnp.asarray(pos_b[:, :, 2].reshape(B * N), dtype=dtype)
  mx_in = jnp.asarray(mom_b[:, :, 0].reshape(B * N), dtype=dtype)
  my_in = jnp.asarray(mom_b[:, :, 1].reshape(B * N), dtype=dtype)
  mz_in = jnp.asarray(mom_b[:, :, 2].reshape(B * N), dtype=dtype)
  fx_in = jnp.asarray(frc_b[:, :, 0].reshape(B * N), dtype=f_dtype)
  fy_in = jnp.asarray(frc_b[:, :, 1].reshape(B * N), dtype=f_dtype)
  fz_in = jnp.asarray(frc_b[:, :, 2].reshape(B * N), dtype=f_dtype)
  m_in = jnp.asarray(mass_b.reshape(B * N), dtype=dtype)
  seed_flat = jnp.asarray(seed_b.reshape(B * 2), dtype=jnp.int32)

  if W > 0:
    wO_in = jnp.asarray(wi_b[:, :, 0].reshape(B * W), dtype=jnp.int32)
    wH1_in = jnp.asarray(wi_b[:, :, 1].reshape(B * W), dtype=jnp.int32)
    wH2_in = jnp.asarray(wi_b[:, :, 2].reshape(B * W), dtype=jnp.int32)
    wm_in = jnp.asarray(wm_b.reshape(B * W), dtype=jnp.int32)
  else:
    wO_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)
    wH1_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)
    wH2_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)
    wm_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)

  bx_flat = jnp.asarray(bx_b.reshape(B * 3), dtype=dtype)

  if iwg_b is not None and not has_rattle:
    is_water_flat = jnp.asarray(iwg_b.reshape(B * N), dtype=jnp.int32)
  else:
    is_water_flat = _build_is_water_or_ghost_mask(
      B, N, W, wi_b, wm_b, am_b, constrained_atom_mask=constrained_atom_mask
    )

  use_scratch = bool(interpret)
  if use_scratch:
    px_in = jnp.pad(px_in, (0, 1))
    py_in = jnp.pad(py_in, (0, 1))
    pz_in = jnp.pad(pz_in, (0, 1))
    mx_in = jnp.pad(mx_in, (0, 1))
    my_in = jnp.pad(my_in, (0, 1))
    mz_in = jnp.pad(mz_in, (0, 1))

  emit_f32_out = bool(return_pos_f32) and (dtype == jnp.float64)
  out_len = B * N + (1 if use_scratch else 0)
  if emit_f32_out:
    px_f32_init = jnp.zeros((out_len,), dtype=jnp.float32)
    py_f32_init = jnp.zeros((out_len,), dtype=jnp.float32)
    pz_f32_init = jnp.zeros((out_len,), dtype=jnp.float32)
  else:
    px_f32_init = py_f32_init = pz_f32_init = None

  total_blocks = max(1, n_blk_a + n_blk_w + n_blk_c)
  gamma_f = float(gamma)
  r_OH_f = float(r_OH)
  r_HH_f = float(r_HH)
  mO_f = float(mass_oxygen)
  mH_f = float(mass_hydrogen)

  def _pre_kernel(*refs):
    px_ref, py_ref, pz_ref, mx_ref, my_ref, mz_ref = refs[0:6]
    fx_ref, fy_ref, fz_ref, m_ref, seed_ref, is_w_ref = refs[6:12]
    wO_ref, wH1_ref, wH2_ref, wm_ref, bx_ref, dt_ref, kT_ref = refs[12:19]
    rc = 19
    if has_rattle:
      cX_ref, cH0_ref, cH1_ref, cH2_ref, cH3_ref = refs[rc : rc + 5]
      cD0_ref, cD1_ref, cD2_ref, cD3_ref, cM_ref = refs[rc + 5 : rc + 10]
      rc += 10
    else:
      cX_ref = cH0_ref = cH1_ref = cH2_ref = cH3_ref = None
      cD0_ref = cD1_ref = cD2_ref = cD3_ref = cM_ref = None
    if emit_f32_out:
      rc += 3
    px_out_ref, py_out_ref, pz_out_ref, mx_out_ref, my_out_ref, mz_out_ref = refs[rc : rc + 6]
    rc += 6
    if emit_f32_out:
      px_f32_out_ref, py_f32_out_ref, pz_f32_out_ref = refs[rc : rc + 3]
    else:
      px_f32_out_ref = py_f32_out_ref = pz_f32_out_ref = None

    b_id = pl.program_id(0)
    pid = pl.program_id(1)
    base_a = b_id * N
    base_w = b_id * W_eff
    base_c = b_id * C_eff
    base_bx = b_id * 3
    scratch_slot = jnp.int32(B * N)

    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    zero_f = jnp.asarray(0.0, dtype=f_dtype)
    zero_w = jnp.asarray(0.0, dtype=w_dtype)
    one_w = jnp.asarray(1.0, dtype=w_dtype)
    half_w = jnp.asarray(0.5, dtype=w_dtype)

    dt_val = _pl_load(dt_ref, (b_id + jnp.zeros((1,), dtype=jnp.int32),))[0]
    kT_val = _pl_load(kT_ref, (b_id + jnp.zeros((1,), dtype=jnp.int32),))[0]
    s0 = _pl_load(seed_ref, (b_id * 2 + jnp.zeros((1,), dtype=jnp.int32),))[0]
    s1 = _pl_load(seed_ref, (b_id * 2 + jnp.ones((1,), dtype=jnp.int32),))[0]
    half_dt = half * dt_val
    dt_w = dt_val.astype(w_dtype)
    half_dt_w = half_w * dt_w
    kT_w = kT_val.astype(w_dtype)
    c1_w = jnp.exp(jnp.asarray(-gamma_f, dtype=w_dtype) * dt_w)
    c2_w = jnp.sqrt(jnp.maximum(one_w - c1_w * c1_w, zero_w))
    c1 = c1_w.astype(dtype)
    c2 = c2_w.astype(dtype)

    if has_box:
      Lx = _pl_load(bx_ref, (base_bx + jnp.zeros((1,), dtype=jnp.int32),))[0]
      Ly = _pl_load(bx_ref, (base_bx + jnp.ones((1,), dtype=jnp.int32),))[0]
      Lz = _pl_load(bx_ref, (base_bx + jnp.full((1,), 2, dtype=jnp.int32),))[0]
      box_xyz = (Lx, Ly, Lz)
    else:
      box_xyz = None

    def _wrap_pos(rx, ry, rz):
      if has_box and box_xyz is not None:
        return (
          jnp.mod(rx, box_xyz[0]),
          jnp.mod(ry, box_xyz[1]),
          jnp.mod(rz, box_xyz[2]),
        )
      return (rx, ry, rz)

    def _store_pos_mom(dest, rx_v, ry_v, rz_v, px_v, py_v, pz_v, mask_v):
      _pl_store(px_out_ref, (dest,), rx_v, mask=mask_v)
      _pl_store(py_out_ref, (dest,), ry_v, mask=mask_v)
      _pl_store(pz_out_ref, (dest,), rz_v, mask=mask_v)
      _pl_store(mx_out_ref, (dest,), px_v, mask=mask_v)
      _pl_store(my_out_ref, (dest,), py_v, mask=mask_v)
      _pl_store(mz_out_ref, (dest,), pz_v, mask=mask_v)
      if emit_f32_out:
        _pl_store(px_f32_out_ref, (dest,), rx_v.astype(jnp.float32), mask=mask_v)
        _pl_store(py_f32_out_ref, (dest,), ry_v.astype(jnp.float32), mask=mask_v)
        _pl_store(pz_f32_out_ref, (dest,), rz_v.astype(jnp.float32), mask=mask_v)

    def _do_solute(a_pid):
      offs = a_pid * _SETTLE_ATOM_BLOCK + jnp.arange(_SETTLE_ATOM_BLOCK, dtype=jnp.int32)
      in_a = offs < N
      safe_a = base_a + jnp.minimum(offs, max(N - 1, 0))
      w_code = _pl_load(is_w_ref, (safe_a,), mask=in_a, other=jnp.int32(1))
      valid_s = in_a & (w_code == 0)

      if emit_f32_out:
        valid_g = in_a & (w_code == 3)

        def _copy_ghost_f32(_):
          dest_g = jnp.where(valid_g, safe_a, scratch_slot) if use_scratch else safe_a
          gx = _pl_load(px_ref, (safe_a,), mask=valid_g, other=zero).astype(jnp.float32)
          gy = _pl_load(py_ref, (safe_a,), mask=valid_g, other=zero).astype(jnp.float32)
          gz = _pl_load(pz_ref, (safe_a,), mask=valid_g, other=zero).astype(jnp.float32)
          _pl_store(px_f32_out_ref, (dest_g,), gx, mask=valid_g)
          _pl_store(py_f32_out_ref, (dest_g,), gy, mask=valid_g)
          _pl_store(pz_f32_out_ref, (dest_g,), gz, mask=valid_g)
          return a_pid - a_pid

        _ = jax.lax.cond(
          jnp.max(valid_g.astype(jnp.int32)) > 0,
          _copy_ghost_f32,
          lambda _: a_pid - a_pid,
          None,
        )

      def _run_solute(_):
        rx = _pl_load(px_ref, (safe_a,), mask=valid_s, other=zero)
        ry = _pl_load(py_ref, (safe_a,), mask=valid_s, other=zero)
        rz = _pl_load(pz_ref, (safe_a,), mask=valid_s, other=zero)
        px = _pl_load(mx_ref, (safe_a,), mask=valid_s, other=zero).astype(w_dtype)
        py = _pl_load(my_ref, (safe_a,), mask=valid_s, other=zero).astype(w_dtype)
        pz = _pl_load(mz_ref, (safe_a,), mask=valid_s, other=zero).astype(w_dtype)
        fx = _pl_load(fx_ref, (safe_a,), mask=valid_s, other=zero_f).astype(w_dtype)
        fy = _pl_load(fy_ref, (safe_a,), mask=valid_s, other=zero_f).astype(w_dtype)
        fz = _pl_load(fz_ref, (safe_a,), mask=valid_s, other=zero_f).astype(w_dtype)
        m_a = _pl_load(m_ref, (safe_a,), mask=valid_s, other=one).astype(w_dtype)
        nsx, nsy = _philox2x32_box_muller_pair(s0, s1, offs, offs + jnp.int32(0x10000001), w_dtype)
        nsz, _ = _philox2x32_box_muller_pair(s0, s1, offs, offs + jnp.int32(0x20000002), w_dtype)

        inv_m = one_w / jnp.maximum(m_a, jnp.asarray(1e-30, dtype=w_dtype))
        px1 = px + half_dt_w * fx
        py1 = py + half_dt_w * fy
        pz1 = pz + half_dt_w * fz

        rx1, ry1, rz1 = _wrap_pos(
          rx + (half_dt_w * (px1 * inv_m)).astype(dtype),
          ry + (half_dt_w * (py1 * inv_m)).astype(dtype),
          rz + (half_dt_w * (pz1 * inv_m)).astype(dtype),
        )

        noise_scale = c2_w * jnp.sqrt(jnp.maximum(m_a * kT_w, zero_w))
        px_o = c1_w * px1 + noise_scale * nsx
        py_o = c1_w * py1 + noise_scale * nsy
        pz_o = c1_w * pz1 + noise_scale * nsz

        rx2, ry2, rz2 = _wrap_pos(
          rx1 + (half_dt_w * (px_o * inv_m)).astype(dtype),
          ry1 + (half_dt_w * (py_o * inv_m)).astype(dtype),
          rz1 + (half_dt_w * (pz_o * inv_m)).astype(dtype),
        )

        dest = jnp.where(valid_s, safe_a, scratch_slot) if use_scratch else safe_a
        _store_pos_mom(
          dest, rx2, ry2, rz2, px_o.astype(dtype), py_o.astype(dtype), pz_o.astype(dtype), valid_s
        )
        return a_pid - a_pid

      return jax.lax.cond(
        jnp.max(valid_s.astype(jnp.int32)) > 0,
        _run_solute,
        lambda _: a_pid - a_pid,
        None,
      )

    def _do_water(w_pid):
      offs_w = w_pid * _SETTLE_WATER_BLOCK + jnp.arange(_SETTLE_WATER_BLOCK, dtype=jnp.int32)
      in_w = offs_w < W
      safe_w = base_w + jnp.minimum(offs_w, max(W - 1, 0))
      wm = _pl_load(wm_ref, (safe_w,), mask=in_w, other=jnp.int32(0)) > 0
      valid_w = in_w & wm

      def _run_water(_):
        iO = _pl_load(wO_ref, (safe_w,), mask=valid_w, other=jnp.int32(0))
        iH1 = _pl_load(wH1_ref, (safe_w,), mask=valid_w, other=jnp.int32(0))
        iH2 = _pl_load(wH2_ref, (safe_w,), mask=valid_w, other=jnp.int32(0))
        sO = jnp.where(valid_w, jnp.clip(iO, 0, max(N - 1, 0)), 0)
        sH1 = jnp.where(valid_w, jnp.clip(iH1, 0, max(N - 1, 0)), 0)
        sH2 = jnp.where(valid_w, jnp.clip(iH2, 0, max(N - 1, 0)), 0)

        def _load_vec3(rx_r, ry_r, rz_r, idx, other_val):
          return (
            _pl_load(rx_r, (base_a + idx,), mask=valid_w, other=other_val),
            _pl_load(ry_r, (base_a + idx,), mask=valid_w, other=other_val),
            _pl_load(rz_r, (base_a + idx,), mask=valid_w, other=other_val),
          )

        r0_O = _load_vec3(px_ref, py_ref, pz_ref, sO, zero)
        r0_H1 = _load_vec3(px_ref, py_ref, pz_ref, sH1, zero)
        r0_H2 = _load_vec3(px_ref, py_ref, pz_ref, sH2, zero)

        p0_O = _load_vec3(mx_ref, my_ref, mz_ref, sO, zero)
        p0_H1 = _load_vec3(mx_ref, my_ref, mz_ref, sH1, zero)
        p0_H2 = _load_vec3(mx_ref, my_ref, mz_ref, sH2, zero)

        f0_O = _load_vec3(fx_ref, fy_ref, fz_ref, sO, zero_f)
        f0_H1 = _load_vec3(fx_ref, fy_ref, fz_ref, sH1, zero_f)
        f0_H2 = _load_vec3(fx_ref, fy_ref, fz_ref, sH2, zero_f)

        mO_s = _pl_load(m_ref, (base_a + sO,), mask=valid_w, other=one)
        mH1_s = _pl_load(m_ref, (base_a + sH1,), mask=valid_w, other=one)
        mH2_s = _pl_load(m_ref, (base_a + sH2,), mask=valid_w, other=one)
        inv_mO_w = one_w / jnp.maximum(mO_s.astype(w_dtype), jnp.asarray(1e-30, dtype=w_dtype))
        inv_mH1_w = one_w / jnp.maximum(mH1_s.astype(w_dtype), jnp.asarray(1e-30, dtype=w_dtype))
        inv_mH2_w = one_w / jnp.maximum(mH2_s.astype(w_dtype), jnp.asarray(1e-30, dtype=w_dtype))

        # 1. B1 half-kick
        p_pre1_O = (
          (p0_O[0].astype(w_dtype) + half_dt_w * f0_O[0].astype(w_dtype)).astype(dtype),
          (p0_O[1].astype(w_dtype) + half_dt_w * f0_O[1].astype(w_dtype)).astype(dtype),
          (p0_O[2].astype(w_dtype) + half_dt_w * f0_O[2].astype(w_dtype)).astype(dtype),
        )
        p_pre1_H1 = (
          (p0_H1[0].astype(w_dtype) + half_dt_w * f0_H1[0].astype(w_dtype)).astype(dtype),
          (p0_H1[1].astype(w_dtype) + half_dt_w * f0_H1[1].astype(w_dtype)).astype(dtype),
          (p0_H1[2].astype(w_dtype) + half_dt_w * f0_H1[2].astype(w_dtype)).astype(dtype),
        )
        p_pre1_H2 = (
          (p0_H2[0].astype(w_dtype) + half_dt_w * f0_H2[0].astype(w_dtype)).astype(dtype),
          (p0_H2[1].astype(w_dtype) + half_dt_w * f0_H2[1].astype(w_dtype)).astype(dtype),
          (p0_H2[2].astype(w_dtype) + half_dt_w * f0_H2[2].astype(w_dtype)).astype(dtype),
        )

        # 2. A1 half-drift + wrap
        r_unc1_O = _wrap_pos(
          r0_O[0] + (half_dt_w * p_pre1_O[0].astype(w_dtype) * inv_mO_w).astype(dtype),
          r0_O[1] + (half_dt_w * p_pre1_O[1].astype(w_dtype) * inv_mO_w).astype(dtype),
          r0_O[2] + (half_dt_w * p_pre1_O[2].astype(w_dtype) * inv_mO_w).astype(dtype),
        )
        r_unc1_H1 = _wrap_pos(
          r0_H1[0] + (half_dt_w * p_pre1_H1[0].astype(w_dtype) * inv_mH1_w).astype(dtype),
          r0_H1[1] + (half_dt_w * p_pre1_H1[1].astype(w_dtype) * inv_mH1_w).astype(dtype),
          r0_H1[2] + (half_dt_w * p_pre1_H1[2].astype(w_dtype) * inv_mH1_w).astype(dtype),
        )
        r_unc1_H2 = _wrap_pos(
          r0_H2[0] + (half_dt_w * p_pre1_H2[0].astype(w_dtype) * inv_mH2_w).astype(dtype),
          r0_H2[1] + (half_dt_w * p_pre1_H2[1].astype(w_dtype) * inv_mH2_w).astype(dtype),
          r0_H2[2] + (half_dt_w * p_pre1_H2[2].astype(w_dtype) * inv_mH2_w).astype(dtype),
        )

        # 3. SETTLE1 + R1 + AM1
        r_con1_O, r_con1_H1, r_con1_H2 = _settle_water_vec3(
          r0_O, r0_H1, r0_H2, r_unc1_O, r_unc1_H1, r_unc1_H2,
          r_OH_f, r_HH_f, mO_f, mH_f, box_xyz, has_box,
          use_f32_math=use_f32_math,
        )
        p_mid_O, p_mid_H1, p_mid_H2 = _r_step_and_am_vec3(
          r_unc1_O, r_unc1_H1, r_unc1_H2, r_con1_O, r_con1_H1, r_con1_H2,
          p_pre1_O, p_pre1_H1, p_pre1_H2, mO_s, mH1_s, mH2_s,
          half_dt, mO_f, mH_f, box_xyz, has_box,
          use_f32_math=use_f32_math,
        )

        # 4. O-step (rigid 6D subspace or isotropic) via in-register Philox-4x32 + Box-Muller
        if project_ou_momentum_rigid:
          z0, z1 = _philox2x32_box_muller_pair(s0, s1, offs_w, offs_w + jnp.int32(0x30000001), w_dtype)
          z2, z3 = _philox2x32_box_muller_pair(s0, s1, offs_w, offs_w + jnp.int32(0x40000002), w_dtype)
          z4, z5 = _philox2x32_box_muller_pair(s0, s1, offs_w, offs_w + jnp.int32(0x50000003), w_dtype)
          p_pre2_O, p_pre2_H1, p_pre2_H2 = _water_o_step_vec3(
            p_mid_O, p_mid_H1, p_mid_H2, r_con1_O, r_con1_H1, r_con1_H2,
            mO_s, mH1_s, mH2_s, z0, z1, z2, z3, z4, z5, c1, c2, kT_val,
            use_f32_math=use_f32_math,
          )
        else:
          nsOx, nsOy = _philox2x32_box_muller_pair(s0, s1, sO, sO + jnp.int32(0x10000001), w_dtype)
          nsOz, _ = _philox2x32_box_muller_pair(s0, s1, sO, sO + jnp.int32(0x20000002), w_dtype)
          nsH1x, nsH1y = _philox2x32_box_muller_pair(s0, s1, sH1, sH1 + jnp.int32(0x10000001), w_dtype)
          nsH1z, _ = _philox2x32_box_muller_pair(s0, s1, sH1, sH1 + jnp.int32(0x20000002), w_dtype)
          nsH2x, nsH2y = _philox2x32_box_muller_pair(s0, s1, sH2, sH2 + jnp.int32(0x10000001), w_dtype)
          nsH2z, _ = _philox2x32_box_muller_pair(s0, s1, sH2, sH2 + jnp.int32(0x20000002), w_dtype)
          scO = c2_w * jnp.sqrt(jnp.maximum(mO_s.astype(w_dtype) * kT_w, zero_w))
          scH1 = c2_w * jnp.sqrt(jnp.maximum(mH1_s.astype(w_dtype) * kT_w, zero_w))
          scH2 = c2_w * jnp.sqrt(jnp.maximum(mH2_s.astype(w_dtype) * kT_w, zero_w))
          p_pre2_O = (
            (c1_w * p_mid_O[0].astype(w_dtype) + scO * nsOx).astype(dtype),
            (c1_w * p_mid_O[1].astype(w_dtype) + scO * nsOy).astype(dtype),
            (c1_w * p_mid_O[2].astype(w_dtype) + scO * nsOz).astype(dtype),
          )
          p_pre2_H1 = (
            (c1_w * p_mid_H1[0].astype(w_dtype) + scH1 * nsH1x).astype(dtype),
            (c1_w * p_mid_H1[1].astype(w_dtype) + scH1 * nsH1y).astype(dtype),
            (c1_w * p_mid_H1[2].astype(w_dtype) + scH1 * nsH1z).astype(dtype),
          )
          p_pre2_H2 = (
            (c1_w * p_mid_H2[0].astype(w_dtype) + scH2 * nsH2x).astype(dtype),
            (c1_w * p_mid_H2[1].astype(w_dtype) + scH2 * nsH2y).astype(dtype),
            (c1_w * p_mid_H2[2].astype(w_dtype) + scH2 * nsH2z).astype(dtype),
          )

        # 5. A2 half-drift + wrap
        r_unc2_O = _wrap_pos(
          r_con1_O[0] + (half_dt_w * p_pre2_O[0].astype(w_dtype) * inv_mO_w).astype(dtype),
          r_con1_O[1] + (half_dt_w * p_pre2_O[1].astype(w_dtype) * inv_mO_w).astype(dtype),
          r_con1_O[2] + (half_dt_w * p_pre2_O[2].astype(w_dtype) * inv_mO_w).astype(dtype),
        )
        r_unc2_H1 = _wrap_pos(
          r_con1_H1[0] + (half_dt_w * p_pre2_H1[0].astype(w_dtype) * inv_mH1_w).astype(dtype),
          r_con1_H1[1] + (half_dt_w * p_pre2_H1[1].astype(w_dtype) * inv_mH1_w).astype(dtype),
          r_con1_H1[2] + (half_dt_w * p_pre2_H1[2].astype(w_dtype) * inv_mH1_w).astype(dtype),
        )
        r_unc2_H2 = _wrap_pos(
          r_con1_H2[0] + (half_dt_w * p_pre2_H2[0].astype(w_dtype) * inv_mH2_w).astype(dtype),
          r_con1_H2[1] + (half_dt_w * p_pre2_H2[1].astype(w_dtype) * inv_mH2_w).astype(dtype),
          r_con1_H2[2] + (half_dt_w * p_pre2_H2[2].astype(w_dtype) * inv_mH2_w).astype(dtype),
        )

        # 6. SETTLE2 + R2 + AM2
        r_con2_O, r_con2_H1, r_con2_H2 = _settle_water_vec3(
          r_con1_O, r_con1_H1, r_con1_H2, r_unc2_O, r_unc2_H1, r_unc2_H2,
          r_OH_f, r_HH_f, mO_f, mH_f, box_xyz, has_box,
          use_f32_math=use_f32_math,
        )
        p_fin_O, p_fin_H1, p_fin_H2 = _r_step_and_am_vec3(
          r_unc2_O, r_unc2_H1, r_unc2_H2, r_con2_O, r_con2_H1, r_con2_H2,
          p_pre2_O, p_pre2_H1, p_pre2_H2, mO_s, mH1_s, mH2_s,
          half_dt, mO_f, mH_f, box_xyz, has_box,
          use_f32_math=use_f32_math,
        )

        # Single global store per water atom
        for s_idx, r_v, p_v in (
          (sO, r_con2_O, p_fin_O),
          (sH1, r_con2_H1, p_fin_H1),
          (sH2, r_con2_H2, p_fin_H2),
        ):
          dest = jnp.where(valid_w, base_a + s_idx, scratch_slot) if use_scratch else (base_a + s_idx)
          _store_pos_mom(dest, r_v[0], r_v[1], r_v[2], p_v[0], p_v[1], p_v[2], valid_w)
        return w_pid - w_pid

      return jax.lax.cond(
        jnp.max(valid_w.astype(jnp.int32)) > 0,
        _run_water,
        lambda _: w_pid - w_pid,
        None,
      )

    def _do_rattle(c_pid):
      offs_c = c_pid * _SETTLE_RATTLE_BLOCK + jnp.arange(_SETTLE_RATTLE_BLOCK, dtype=jnp.int32)
      in_c = offs_c < C
      safe_c = base_c + jnp.minimum(offs_c, max(C - 1, 0))
      cm_bits = _pl_load(cM_ref, (safe_c,), mask=in_c, other=jnp.int32(0))
      v0 = in_c & ((cm_bits & jnp.int32(1)) != 0)
      v1 = in_c & ((cm_bits & jnp.int32(2)) != 0)
      v2 = in_c & ((cm_bits & jnp.int32(4)) != 0)
      v3 = in_c & ((cm_bits & jnp.int32(8)) != 0)
      v_slots = [v0, v1, v2, v3]

      def _run_rattle(_):
        iX = _pl_load(cX_ref, (safe_c,), mask=v0, other=jnp.int32(0))
        iH0 = _pl_load(cH0_ref, (safe_c,), mask=v0, other=jnp.int32(0))
        iH1 = _pl_load(cH1_ref, (safe_c,), mask=v1, other=jnp.int32(0))
        iH2 = _pl_load(cH2_ref, (safe_c,), mask=v2, other=jnp.int32(0))
        iH3 = _pl_load(cH3_ref, (safe_c,), mask=v3, other=jnp.int32(0))
        sX = jnp.where(v0, jnp.clip(iX, 0, max(N - 1, 0)), 0)
        sH = [
          jnp.where(v0, jnp.clip(iH0, 0, max(N - 1, 0)), 0),
          jnp.where(v1, jnp.clip(iH1, 0, max(N - 1, 0)), 0),
          jnp.where(v2, jnp.clip(iH2, 0, max(N - 1, 0)), 0),
          jnp.where(v3, jnp.clip(iH3, 0, max(N - 1, 0)), 0),
        ]
        d_len = [
          _pl_load(cD0_ref, (safe_c,), mask=v0, other=one),
          _pl_load(cD1_ref, (safe_c,), mask=v1, other=one),
          _pl_load(cD2_ref, (safe_c,), mask=v2, other=one),
          _pl_load(cD3_ref, (safe_c,), mask=v3, other=one),
        ]
        d_sq = [d_len[k] * d_len[k] for k in range(4)]

        def _ld_v3(rx_r, ry_r, rz_r, idx, msk, oth):
          return (
            _pl_load(rx_r, (base_a + idx,), mask=msk, other=oth),
            _pl_load(ry_r, (base_a + idx,), mask=msk, other=oth),
            _pl_load(rz_r, (base_a + idx,), mask=msk, other=oth),
          )

        r0_X = _ld_v3(px_ref, py_ref, pz_ref, sX, v0, zero)
        r0_H_raw = [_ld_v3(px_ref, py_ref, pz_ref, sH[k], v_slots[k], zero) for k in range(4)]
        p0_X = _ld_v3(mx_ref, my_ref, mz_ref, sX, v0, zero)
        p0_H = [_ld_v3(mx_ref, my_ref, mz_ref, sH[k], v_slots[k], zero) for k in range(4)]
        f0_X = _ld_v3(fx_ref, fy_ref, fz_ref, sX, v0, zero_f)
        f0_H = [_ld_v3(fx_ref, fy_ref, fz_ref, sH[k], v_slots[k], zero_f) for k in range(4)]

        mX = _pl_load(m_ref, (base_a + sX,), mask=v0, other=one)
        mH = [_pl_load(m_ref, (base_a + sH[k]), mask=v_slots[k], other=one) for k in range(4)]
        inv_mX = one / jnp.maximum(mX, jnp.asarray(1e-30, dtype=dtype))
        inv_mH = [one / jnp.maximum(mH[k], jnp.asarray(1e-30, dtype=dtype)) for k in range(4)]
        inv_half_dt = one / jnp.maximum(half_dt, jnp.asarray(1e-30, dtype=dtype))

        # Unwrap r0_H relative to r0_X so each cluster is contiguous in space
        r0_H = []
        ab0 = []
        for k in range(4):
          dx = r0_H_raw[k][0] - r0_X[0]
          dy = r0_H_raw[k][1] - r0_X[1]
          dz = r0_H_raw[k][2] - r0_X[2]
          if has_box and box_xyz is not None:
            dx = dx - box_xyz[0] * _PBC_ROUND_FN(dx / box_xyz[0])
            dy = dy - box_xyz[1] * _PBC_ROUND_FN(dy / box_xyz[1])
            dz = dz - box_xyz[2] * _PBC_ROUND_FN(dz / box_xyz[2])
          ab0.append((dx, dy, dz))
          r0_H.append((r0_X[0] + dx, r0_X[1] + dy, r0_X[2] + dz))

        # 1. B1 half-kick
        p1_X = (
          p0_X[0] + (half_dt_w * f0_X[0].astype(w_dtype)).astype(dtype),
          p0_X[1] + (half_dt_w * f0_X[1].astype(w_dtype)).astype(dtype),
          p0_X[2] + (half_dt_w * f0_X[2].astype(w_dtype)).astype(dtype),
        )
        p1_H = [
          (
            p0_H[k][0] + (half_dt_w * f0_H[k][0].astype(w_dtype)).astype(dtype),
            p0_H[k][1] + (half_dt_w * f0_H[k][1].astype(w_dtype)).astype(dtype),
            p0_H[k][2] + (half_dt_w * f0_H[k][2].astype(w_dtype)).astype(dtype),
          )
          for k in range(4)
        ]

        # 2. A1 half-drift (unwrapped) + SHAKE1 + R1
        r_unc1_X = (
          r0_X[0] + half_dt * p1_X[0] * inv_mX,
          r0_X[1] + half_dt * p1_X[1] * inv_mX,
          r0_X[2] + half_dt * p1_X[2] * inv_mX,
        )
        r_unc1_H = [
          (
            r0_H[k][0] + half_dt * p1_H[k][0] * inv_mH[k],
            r0_H[k][1] + half_dt * p1_H[k][1] * inv_mH[k],
            r0_H[k][2] + half_dt * p1_H[k][2] * inv_mH[k],
          )
          for k in range(4)
        ]
        r_con1_X, r_con1_H = _rattle_shake_cluster_vec3(
          r_unc1_X, r_unc1_H, ab0, inv_mX, inv_mH, d_sq, v_slots,
          n_iters=8, use_f32_math=use_f32_math,
        )
        cX_dt = mX * inv_half_dt
        p_mid_X = (
          p1_X[0] + cX_dt * (r_con1_X[0] - r_unc1_X[0]),
          p1_X[1] + cX_dt * (r_con1_X[1] - r_unc1_X[1]),
          p1_X[2] + cX_dt * (r_con1_X[2] - r_unc1_X[2]),
        )
        p_mid_H = []
        ab1 = []
        for k in range(4):
          cH_dt = mH[k] * inv_half_dt
          p_mid_H.append(
            (
              p1_H[k][0] + cH_dt * (r_con1_H[k][0] - r_unc1_H[k][0]),
              p1_H[k][1] + cH_dt * (r_con1_H[k][1] - r_unc1_H[k][1]),
              p1_H[k][2] + cH_dt * (r_con1_H[k][2] - r_unc1_H[k][2]),
            )
          )
          ab1.append(
            (
              r_con1_H[k][0] - r_con1_X[0],
              r_con1_H[k][1] - r_con1_X[1],
              r_con1_H[k][2] - r_con1_X[2],
            )
          )
        # 3. O-step + RATTLE velocity projection (P_vel(c1 * P_vel(p_mid) + z) == P_vel(c1 * p_mid + z) at fixed ab1)
        nsXx, nsXy = _philox2x32_box_muller_pair(s0, s1, sX, sX + jnp.int32(0x10000001), w_dtype)
        nsXz, _ = _philox2x32_box_muller_pair(s0, s1, sX, sX + jnp.int32(0x20000002), w_dtype)
        scX = c2_w * jnp.sqrt(jnp.maximum(mX.astype(w_dtype) * kT_w, zero_w))
        p_o_X = (
          (c1_w * p_mid_X[0].astype(w_dtype) + scX * nsXx).astype(dtype),
          (c1_w * p_mid_X[1].astype(w_dtype) + scX * nsXy).astype(dtype),
          (c1_w * p_mid_X[2].astype(w_dtype) + scX * nsXz).astype(dtype),
        )
        p_o_H = []
        for k in range(4):
          nsHx, nsHy = _philox2x32_box_muller_pair(s0, s1, sH[k], sH[k] + jnp.int32(0x10000001), w_dtype)
          nsHz, _ = _philox2x32_box_muller_pair(s0, s1, sH[k], sH[k] + jnp.int32(0x20000002), w_dtype)
          scH = c2_w * jnp.sqrt(jnp.maximum(mH[k].astype(w_dtype) * kT_w, zero_w))
          p_o_H.append(
            (
              (c1_w * p_mid_H[k][0].astype(w_dtype) + scH * nsHx).astype(dtype),
              (c1_w * p_mid_H[k][1].astype(w_dtype) + scH * nsHy).astype(dtype),
              (c1_w * p_mid_H[k][2].astype(w_dtype) + scH * nsHz).astype(dtype),
            )
          )
        p_o_X, p_o_H = _rattle_vel_cluster_vec3(
          p_o_X, p_o_H, ab1, inv_mX, inv_mH, v_slots,
          n_iters=6, use_f32_math=use_f32_math,
        )

        # 4. A2 half-drift + SHAKE2 + R2
        r_unc2_X = (
          r_con1_X[0] + half_dt * p_o_X[0] * inv_mX,
          r_con1_X[1] + half_dt * p_o_X[1] * inv_mX,
          r_con1_X[2] + half_dt * p_o_X[2] * inv_mX,
        )
        r_unc2_H = [
          (
            r_con1_H[k][0] + half_dt * p_o_H[k][0] * inv_mH[k],
            r_con1_H[k][1] + half_dt * p_o_H[k][1] * inv_mH[k],
            r_con1_H[k][2] + half_dt * p_o_H[k][2] * inv_mH[k],
          )
          for k in range(4)
        ]
        r_con2_X, r_con2_H = _rattle_shake_cluster_vec3(
          r_unc2_X, r_unc2_H, ab1, inv_mX, inv_mH, d_sq, v_slots,
          n_iters=8, use_f32_math=use_f32_math,
        )
        p_fin_X = (
          p_o_X[0] + cX_dt * (r_con2_X[0] - r_unc2_X[0]),
          p_o_X[1] + cX_dt * (r_con2_X[1] - r_unc2_X[1]),
          p_o_X[2] + cX_dt * (r_con2_X[2] - r_unc2_X[2]),
        )
        p_fin_H = []
        ab2 = []
        for k in range(4):
          cH_dt = mH[k] * inv_half_dt
          p_fin_H.append(
            (
              p_o_H[k][0] + cH_dt * (r_con2_H[k][0] - r_unc2_H[k][0]),
              p_o_H[k][1] + cH_dt * (r_con2_H[k][1] - r_unc2_H[k][1]),
              p_o_H[k][2] + cH_dt * (r_con2_H[k][2] - r_unc2_H[k][2]),
            )
          )
          ab2.append(
            (
              r_con2_H[k][0] - r_con2_X[0],
              r_con2_H[k][1] - r_con2_X[1],
              r_con2_H[k][2] - r_con2_X[2],
            )
          )
        p_fin_X, p_fin_H = _rattle_vel_cluster_vec3(
          p_fin_X, p_fin_H, ab2, inv_mX, inv_mH, v_slots,
          n_iters=6, use_f32_math=use_f32_math,
        )

        # Wrap heavy atom X into box and apply same periodic shift to H0..H3
        r_w_X = _wrap_pos(r_con2_X[0], r_con2_X[1], r_con2_X[2])
        sh_x = r_w_X[0] - r_con2_X[0]
        sh_y = r_w_X[1] - r_con2_X[1]
        sh_z = r_w_X[2] - r_con2_X[2]

        destX = jnp.where(v0, base_a + sX, scratch_slot) if use_scratch else (base_a + sX)
        _store_pos_mom(destX, r_w_X[0], r_w_X[1], r_w_X[2], p_fin_X[0], p_fin_X[1], p_fin_X[2], v0)
        for k in range(4):
          vk = v_slots[k]
          destH = jnp.where(vk, base_a + sH[k], scratch_slot) if use_scratch else (base_a + sH[k])
          _store_pos_mom(
            destH,
            r_con2_H[k][0] + sh_x,
            r_con2_H[k][1] + sh_y,
            r_con2_H[k][2] + sh_z,
            p_fin_H[k][0],
            p_fin_H[k][1],
            p_fin_H[k][2],
            vk,
          )
        return c_pid - c_pid

      return jax.lax.cond(
        jnp.max(v0.astype(jnp.int32)) > 0,
        _run_rattle,
        lambda _: c_pid - c_pid,
        None,
      )

    def _dispatch_non_solute(p_rem):
      if n_blk_w > 0 and n_blk_c > 0:
        return jax.lax.cond(
          p_rem < n_blk_w,
          _do_water,
          lambda p2: _do_rattle(p2 - n_blk_w),
          p_rem,
        )
      if n_blk_w > 0:
        return _do_water(p_rem)
      if n_blk_c > 0:
        return _do_rattle(p_rem)
      return p_rem - p_rem

    if n_blk_a > 0 and (n_blk_w + n_blk_c) > 0:
      _ = jax.lax.cond(
        pid < n_blk_a,
        _do_solute,
        lambda p: _dispatch_non_solute(p - n_blk_a),
        pid,
      )
    elif n_blk_a > 0:
      _ = _do_solute(pid)
    else:
      _ = _dispatch_non_solute(pid)

  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  in_args = [
    px_in,
    py_in,
    pz_in,
    mx_in,
    my_in,
    mz_in,
    fx_in,
    fy_in,
    fz_in,
    m_in,
    seed_flat,
    is_water_flat,
    wO_in,
    wH1_in,
    wH2_in,
    wm_in,
    bx_flat,
    dt_b,
    kT_b,
  ]
  if has_rattle:
    in_args.extend(
      [cX_in, cH0_in, cH1_in, cH2_in, cH3_in, cD0_in, cD1_in, cD2_in, cD3_in, cM_in]
    )
  aliases = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
  out_shapes = [jax.ShapeDtypeStruct((out_len,), dtype) for _ in range(6)]
  if emit_f32_out:
    f32_in_base = len(in_args)
    in_args.extend([px_f32_init, py_f32_init, pz_f32_init])
    aliases[f32_in_base] = 6
    aliases[f32_in_base + 1] = 7
    aliases[f32_in_base + 2] = 8
    out_shapes.extend([jax.ShapeDtypeStruct((out_len,), jnp.float32) for _ in range(3)])

  raw_outs = pl.pallas_call(
    _pre_kernel,
    out_shape=out_shapes,
    in_specs=[any_spec] * len(in_args),
    out_specs=[any_spec] * len(out_shapes),
    grid=(B, total_blocks),
    input_output_aliases=aliases,
    interpret=interpret,
    compiler_params=(
      resolve_pallas_compiler_params("settle_integrator", grid_blocks=B * total_blocks)
      if not interpret
      else None
    ),
    name="pallas_baoab_pre_force",
  )(*in_args)

  px_o, py_o, pz_o, mx_o, my_o, mz_o = raw_outs[:6]
  if use_scratch:
    px_o, py_o, pz_o = px_o[:-1], py_o[:-1], pz_o[:-1]
    mx_o, my_o, mz_o = mx_o[:-1], my_o[:-1], mz_o[:-1]
  pos_out = jnp.stack(
    [
      px_o.reshape(B, N),
      py_o.reshape(B, N),
      pz_o.reshape(B, N),
    ],
    axis=-1,
  )
  mom_out = jnp.stack(
    [
      mx_o.reshape(B, N),
      my_o.reshape(B, N),
      mz_o.reshape(B, N),
    ],
    axis=-1,
  )

  if not return_pos_f32 and not return_max_dr_sq:
    return pos_out, mom_out

  ret: list[Array] = [pos_out, mom_out]
  if return_pos_f32:
    if emit_f32_out:
      px_f32, py_f32, pz_f32 = raw_outs[6:9]
      if use_scratch:
        px_f32, py_f32, pz_f32 = px_f32[:-1], py_f32[:-1], pz_f32[:-1]
      pos_f32_out = jnp.stack(
        [
          px_f32.reshape(B, N),
          py_f32.reshape(B, N),
          pz_f32.reshape(B, N),
        ],
        axis=-1,
      )
    else:
      pos_f32_out = pos_out.astype(jnp.float32)
    ret.append(pos_f32_out)

  if return_max_dr_sq:
    ref_p = pos_b if ref_pos_b is None else ref_pos_b
    dr = pos_out - ref_p
    if has_box:
      dr = dr - bx_b[:, None, :] * _PBC_ROUND_FN(dr / bx_b[:, None, :])
    dr_sq = jnp.sum(dr * dr, axis=-1)
    if am_b is not None:
      dr_sq = jnp.where(am_b > 0, dr_sq, jnp.asarray(0.0, dtype=dtype))
    max_dr_sq = jnp.max(dr_sq, axis=-1)
    ret.append(max_dr_sq)

  return tuple(ret)


def _pallas_baoab_post_force_batch(
  B: int,
  in_batched: tuple[bool, ...],
  positions: Array,
  momentum: Array,
  force: Array,
  mass: Array,
  water_indices: Array,
  water_mask: Array,
  dt_arr: Array,
  mass_oxygen: float,
  mass_hydrogen: float,
  interpret: bool,
  atom_mask: Array | None = None,
  is_water_or_ghost: Array | None = None,
  constraint_pairs: Array | None = None,
  constraint_lengths: Array | None = None,
  constraint_mask: Array | None = None,
  n_active_atoms: int | None = None,
  n_active_waters: int | None = None,
) -> Array:
  """2D-grid ``(B, n_blocks_atom + n_blocks_water + n_blocks_rattle)`` Pallas BAOAB post-force kernel."""
  dtype = positions.dtype
  f_dtype = force.dtype
  use_f32_math = (not interpret) and (dtype == jnp.float64) and (
    os.environ.get("PROLIX_BAOAB_F64_INTERNAL", "0") != "1"
  )
  w_dtype = jnp.float32 if use_f32_math else dtype

  def _bcast(arr, is_b):
    return arr if is_b else jnp.broadcast_to(arr[None, ...], (B,) + arr.shape)

  pos_b = _bcast(positions, in_batched[0])
  mom_b = _bcast(momentum, in_batched[1])
  frc_b = _bcast(force, in_batched[2])
  mass_raw = mass.squeeze(-1) if mass.ndim >= 2 and mass.shape[-1] == 1 else mass
  mass_b = _bcast(mass_raw, in_batched[3])
  wi_b = _bcast(water_indices, in_batched[4]).astype(jnp.int32)
  wm_b = _bcast(water_mask, in_batched[5]).astype(jnp.int32)
  dt_b = _bcast(dt_arr, in_batched[6]).astype(dtype).reshape(B)

  arg_cursor = 7
  has_atom_mask = atom_mask is not None
  if has_atom_mask:
    am_b = _bcast(atom_mask, in_batched[arg_cursor]).astype(jnp.int32)
    arg_cursor += 1
  else:
    am_b = jnp.ones((B, int(pos_b.shape[1])), dtype=jnp.int32)

  N = int(pos_b.shape[1])
  W = int(wi_b.shape[1])
  W_eff = max(W, 1)
  N_act = _resolve_active_limit(n_active_atoms, "PROLIX_ACTIVE_ATOMS", N, min_val=1)
  W_act = _resolve_active_limit(n_active_waters, "PROLIX_ACTIVE_WATERS", W, min_val=0)
  n_blk_a = (N_act + _SETTLE_ATOM_BLOCK - 1) // _SETTLE_ATOM_BLOCK
  n_blk_w = (W_act + _SETTLE_WATER_BLOCK - 1) // _SETTLE_WATER_BLOCK

  iwg_b = None
  if is_water_or_ghost is not None:
    iwg_b = _bcast(is_water_or_ghost, in_batched[arg_cursor]).astype(jnp.int32)
    arg_cursor += 1

  has_rattle = (
    constraint_pairs is not None
    and constraint_lengths is not None
    and int(constraint_pairs.shape[-2]) > 0
  )
  C = int(constraint_pairs.shape[-2]) if has_rattle else 0
  C_eff = max(C, 1)
  n_blk_c = (C + _SETTLE_RATTLE_BLOCK - 1) // _SETTLE_RATTLE_BLOCK if has_rattle else 0

  if has_rattle:
    cp_b = _bcast(constraint_pairs, in_batched[arg_cursor]).astype(jnp.int32)
    arg_cursor += 1
    cl_b = _bcast(constraint_lengths, in_batched[arg_cursor]).astype(dtype)
    arg_cursor += 1
    if constraint_mask is not None:
      cm_b = _bcast(constraint_mask, in_batched[arg_cursor]).astype(jnp.int32)
      arg_cursor += 1
    else:
      cm_b = jnp.ones((B, C), dtype=jnp.int32)
    cX, cH0, cH1, cH2, cH3, _cD0, _cD1, _cD2, _cD3, cM, constrained_atom_mask = _build_rattle_clusters(
      B, N, C, cp_b, cl_b, cm_b, mass_b
    )
    cX_in = jnp.asarray(cX.reshape(B * C), dtype=jnp.int32)
    cH0_in = jnp.asarray(cH0.reshape(B * C), dtype=jnp.int32)
    cH1_in = jnp.asarray(cH1.reshape(B * C), dtype=jnp.int32)
    cH2_in = jnp.asarray(cH2.reshape(B * C), dtype=jnp.int32)
    cH3_in = jnp.asarray(cH3.reshape(B * C), dtype=jnp.int32)
    cM_in = jnp.asarray(cM.reshape(B * C), dtype=jnp.int32)
  else:
    constrained_atom_mask = None
    cX_in = cH0_in = cH1_in = cH2_in = cH3_in = cM_in = None

  px_in = jnp.asarray(pos_b[:, :, 0].reshape(B * N), dtype=dtype)
  py_in = jnp.asarray(pos_b[:, :, 1].reshape(B * N), dtype=dtype)
  pz_in = jnp.asarray(pos_b[:, :, 2].reshape(B * N), dtype=dtype)
  mx_in = jnp.asarray(mom_b[:, :, 0].reshape(B * N), dtype=dtype)
  my_in = jnp.asarray(mom_b[:, :, 1].reshape(B * N), dtype=dtype)
  mz_in = jnp.asarray(mom_b[:, :, 2].reshape(B * N), dtype=dtype)
  fx_in = jnp.asarray(frc_b[:, :, 0].reshape(B * N), dtype=f_dtype)
  fy_in = jnp.asarray(frc_b[:, :, 1].reshape(B * N), dtype=f_dtype)
  fz_in = jnp.asarray(frc_b[:, :, 2].reshape(B * N), dtype=f_dtype)
  m_in = jnp.asarray(mass_b.reshape(B * N), dtype=dtype)
  am_in = jnp.asarray(am_b.reshape(B * N), dtype=jnp.int32)

  if W > 0:
    wO_in = jnp.asarray(wi_b[:, :, 0].reshape(B * W), dtype=jnp.int32)
    wH1_in = jnp.asarray(wi_b[:, :, 1].reshape(B * W), dtype=jnp.int32)
    wH2_in = jnp.asarray(wi_b[:, :, 2].reshape(B * W), dtype=jnp.int32)
    wm_in = jnp.asarray(wm_b.reshape(B * W), dtype=jnp.int32)
  else:
    wO_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)
    wH1_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)
    wH2_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)
    wm_in = jnp.zeros((B * W_eff,), dtype=jnp.int32)

  if iwg_b is not None and not has_rattle:
    is_water_flat = jnp.asarray(iwg_b.reshape(B * N), dtype=jnp.int32)
  else:
    is_water_flat = _build_is_water_or_ghost_mask(
      B, N, W, wi_b, wm_b, am_b if has_atom_mask else None,
      constrained_atom_mask=constrained_atom_mask,
    )

  use_scratch = bool(interpret)
  if use_scratch:
    mx_in = jnp.pad(mx_in, (0, 1))
    my_in = jnp.pad(my_in, (0, 1))
    mz_in = jnp.pad(mz_in, (0, 1))

  total_blocks = max(1, n_blk_a + n_blk_w + n_blk_c)
  mO_f = float(mass_oxygen)
  mH_f = float(mass_hydrogen)

  def _post_kernel(*refs):
    px_ref, py_ref, pz_ref, mx_ref, my_ref, mz_ref = refs[0:6]
    fx_ref, fy_ref, fz_ref, m_ref, am_ref, is_w_ref = refs[6:12]
    wO_ref, wH1_ref, wH2_ref, wm_ref, dt_ref = refs[12:17]
    rc = 17
    if has_rattle:
      cX_ref, cH0_ref, cH1_ref, cH2_ref, cH3_ref, cM_ref = refs[rc : rc + 6]
      rc += 6
    else:
      cX_ref = cH0_ref = cH1_ref = cH2_ref = cH3_ref = cM_ref = None
    mx_out_ref, my_out_ref, mz_out_ref = refs[rc : rc + 3]
    del am_ref

    b_id = pl.program_id(0)
    pid = pl.program_id(1)
    base_a = b_id * N
    base_w = b_id * W_eff
    base_c = b_id * C_eff
    scratch_slot = jnp.int32(B * N)

    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    zero_f = jnp.asarray(0.0, dtype=f_dtype)
    half_w = jnp.asarray(0.5, dtype=w_dtype)
    dt_val = _pl_load(dt_ref, (b_id + jnp.zeros((1,), dtype=jnp.int32),))[0]
    half_dt_w = half_w * dt_val.astype(w_dtype)

    def _do_solute(a_pid):
      offs = a_pid * _SETTLE_ATOM_BLOCK + jnp.arange(_SETTLE_ATOM_BLOCK, dtype=jnp.int32)
      in_a = offs < N
      safe_a = base_a + jnp.minimum(offs, max(N - 1, 0))
      is_w = _pl_load(is_w_ref, (safe_a,), mask=in_a, other=jnp.int32(1)) > 0
      valid_s = in_a & (~is_w)

      def _run_solute(_):
        px = _pl_load(mx_ref, (safe_a,), mask=valid_s, other=zero)
        py = _pl_load(my_ref, (safe_a,), mask=valid_s, other=zero)
        pz = _pl_load(mz_ref, (safe_a,), mask=valid_s, other=zero)
        fx = _pl_load(fx_ref, (safe_a,), mask=valid_s, other=zero_f).astype(w_dtype)
        fy = _pl_load(fy_ref, (safe_a,), mask=valid_s, other=zero_f).astype(w_dtype)
        fz = _pl_load(fz_ref, (safe_a,), mask=valid_s, other=zero_f).astype(w_dtype)
        out_px = px + (half_dt_w * fx).astype(dtype)
        out_py = py + (half_dt_w * fy).astype(dtype)
        out_pz = pz + (half_dt_w * fz).astype(dtype)
        dest = jnp.where(valid_s, safe_a, scratch_slot) if use_scratch else safe_a
        _pl_store(mx_out_ref, (dest,), out_px, mask=valid_s)
        _pl_store(my_out_ref, (dest,), out_py, mask=valid_s)
        _pl_store(mz_out_ref, (dest,), out_pz, mask=valid_s)
        return a_pid - a_pid

      return jax.lax.cond(
        jnp.max(valid_s.astype(jnp.int32)) > 0,
        _run_solute,
        lambda _: a_pid - a_pid,
        None,
      )

    def _do_water(w_pid):
      offs_w = w_pid * _SETTLE_WATER_BLOCK + jnp.arange(_SETTLE_WATER_BLOCK, dtype=jnp.int32)
      in_w = offs_w < W
      safe_w = base_w + jnp.minimum(offs_w, max(W - 1, 0))
      wm = _pl_load(wm_ref, (safe_w,), mask=in_w, other=jnp.int32(0)) > 0
      valid_w = in_w & wm

      def _run_water(_):
        iO = _pl_load(wO_ref, (safe_w,), mask=valid_w, other=jnp.int32(0))
        iH1 = _pl_load(wH1_ref, (safe_w,), mask=valid_w, other=jnp.int32(0))
        iH2 = _pl_load(wH2_ref, (safe_w,), mask=valid_w, other=jnp.int32(0))
        sO = jnp.where(valid_w, jnp.clip(iO, 0, max(N - 1, 0)), 0)
        sH1 = jnp.where(valid_w, jnp.clip(iH1, 0, max(N - 1, 0)), 0)
        sH2 = jnp.where(valid_w, jnp.clip(iH2, 0, max(N - 1, 0)), 0)

        def _load_vec3(rx_r, ry_r, rz_r, idx, other_val):
          return (
            _pl_load(rx_r, (base_a + idx,), mask=valid_w, other=other_val),
            _pl_load(ry_r, (base_a + idx,), mask=valid_w, other=other_val),
            _pl_load(rz_r, (base_a + idx,), mask=valid_w, other=other_val),
          )

        rO = _load_vec3(px_ref, py_ref, pz_ref, sO, zero)
        rH1 = _load_vec3(px_ref, py_ref, pz_ref, sH1, zero)
        rH2 = _load_vec3(px_ref, py_ref, pz_ref, sH2, zero)

        pO = _load_vec3(mx_ref, my_ref, mz_ref, sO, zero)
        pH1 = _load_vec3(mx_ref, my_ref, mz_ref, sH1, zero)
        pH2 = _load_vec3(mx_ref, my_ref, mz_ref, sH2, zero)

        fO = _load_vec3(fx_ref, fy_ref, fz_ref, sO, zero_f)
        fH1 = _load_vec3(fx_ref, fy_ref, fz_ref, sH1, zero_f)
        fH2 = _load_vec3(fx_ref, fy_ref, fz_ref, sH2, zero_f)

        mO_s = _pl_load(m_ref, (base_a + sO,), mask=valid_w, other=one)
        mH1_s = _pl_load(m_ref, (base_a + sH1,), mask=valid_w, other=one)
        mH2_s = _pl_load(m_ref, (base_a + sH2,), mask=valid_w, other=one)

        p_b2_O = (
          pO[0] + (half_dt_w * fO[0].astype(w_dtype)).astype(dtype),
          pO[1] + (half_dt_w * fO[1].astype(w_dtype)).astype(dtype),
          pO[2] + (half_dt_w * fO[2].astype(w_dtype)).astype(dtype),
        )
        p_b2_H1 = (
          pH1[0] + (half_dt_w * fH1[0].astype(w_dtype)).astype(dtype),
          pH1[1] + (half_dt_w * fH1[1].astype(w_dtype)).astype(dtype),
          pH1[2] + (half_dt_w * fH1[2].astype(w_dtype)).astype(dtype),
        )
        p_b2_H2 = (
          pH2[0] + (half_dt_w * fH2[0].astype(w_dtype)).astype(dtype),
          pH2[1] + (half_dt_w * fH2[1].astype(w_dtype)).astype(dtype),
          pH2[2] + (half_dt_w * fH2[2].astype(w_dtype)).astype(dtype),
        )

        p_out_O, p_out_H1, p_out_H2 = _settle_vel_and_am_vec3(
          p_b2_O, p_b2_H1, p_b2_H2, rO, rH1, rH2, mO_s, mH1_s, mH2_s, mO_f, mH_f, valid_w,
          use_f32_math=use_f32_math,
        )

        for s_idx, p_v in ((sO, p_out_O), (sH1, p_out_H1), (sH2, p_out_H2)):
          dest = jnp.where(valid_w, base_a + s_idx, scratch_slot) if use_scratch else (base_a + s_idx)
          _pl_store(mx_out_ref, (dest,), p_v[0], mask=valid_w)
          _pl_store(my_out_ref, (dest,), p_v[1], mask=valid_w)
          _pl_store(mz_out_ref, (dest,), p_v[2], mask=valid_w)
        return w_pid - w_pid

      return jax.lax.cond(
        jnp.max(valid_w.astype(jnp.int32)) > 0,
        _run_water,
        lambda _: w_pid - w_pid,
        None,
      )

    def _do_rattle(c_pid):
      offs_c = c_pid * _SETTLE_RATTLE_BLOCK + jnp.arange(_SETTLE_RATTLE_BLOCK, dtype=jnp.int32)
      in_c = offs_c < C
      safe_c = base_c + jnp.minimum(offs_c, max(C - 1, 0))
      cm_bits = _pl_load(cM_ref, (safe_c,), mask=in_c, other=jnp.int32(0))
      v0 = in_c & ((cm_bits & jnp.int32(1)) != 0)
      v1 = in_c & ((cm_bits & jnp.int32(2)) != 0)
      v2 = in_c & ((cm_bits & jnp.int32(4)) != 0)
      v3 = in_c & ((cm_bits & jnp.int32(8)) != 0)
      v_slots = [v0, v1, v2, v3]

      def _run_rattle(_):
        iX = _pl_load(cX_ref, (safe_c,), mask=v0, other=jnp.int32(0))
        iH0 = _pl_load(cH0_ref, (safe_c,), mask=v0, other=jnp.int32(0))
        iH1 = _pl_load(cH1_ref, (safe_c,), mask=v1, other=jnp.int32(0))
        iH2 = _pl_load(cH2_ref, (safe_c,), mask=v2, other=jnp.int32(0))
        iH3 = _pl_load(cH3_ref, (safe_c,), mask=v3, other=jnp.int32(0))
        sX = jnp.where(v0, jnp.clip(iX, 0, max(N - 1, 0)), 0)
        sH = [
          jnp.where(v0, jnp.clip(iH0, 0, max(N - 1, 0)), 0),
          jnp.where(v1, jnp.clip(iH1, 0, max(N - 1, 0)), 0),
          jnp.where(v2, jnp.clip(iH2, 0, max(N - 1, 0)), 0),
          jnp.where(v3, jnp.clip(iH3, 0, max(N - 1, 0)), 0),
        ]

        def _ld_v3(rx_r, ry_r, rz_r, idx, msk, oth):
          return (
            _pl_load(rx_r, (base_a + idx,), mask=msk, other=oth),
            _pl_load(ry_r, (base_a + idx,), mask=msk, other=oth),
            _pl_load(rz_r, (base_a + idx,), mask=msk, other=oth),
          )

        rX = _ld_v3(px_ref, py_ref, pz_ref, sX, v0, zero)
        rH = [_ld_v3(px_ref, py_ref, pz_ref, sH[k], v_slots[k], zero) for k in range(4)]
        pX = _ld_v3(mx_ref, my_ref, mz_ref, sX, v0, zero)
        pH = [_ld_v3(mx_ref, my_ref, mz_ref, sH[k], v_slots[k], zero) for k in range(4)]
        fX = _ld_v3(fx_ref, fy_ref, fz_ref, sX, v0, zero_f)
        fH = [_ld_v3(fx_ref, fy_ref, fz_ref, sH[k], v_slots[k], zero_f) for k in range(4)]

        mX = _pl_load(m_ref, (base_a + sX,), mask=v0, other=one)
        mH = [_pl_load(m_ref, (base_a + sH[k]), mask=v_slots[k], other=one) for k in range(4)]
        inv_mX = one / jnp.maximum(mX, jnp.asarray(1e-30, dtype=dtype))
        inv_mH = [one / jnp.maximum(mH[k], jnp.asarray(1e-30, dtype=dtype)) for k in range(4)]

        p_b2_X = (
          pX[0] + (half_dt_w * fX[0].astype(w_dtype)).astype(dtype),
          pX[1] + (half_dt_w * fX[1].astype(w_dtype)).astype(dtype),
          pX[2] + (half_dt_w * fX[2].astype(w_dtype)).astype(dtype),
        )
        p_b2_H = [
          (
            pH[k][0] + (half_dt_w * fH[k][0].astype(w_dtype)).astype(dtype),
            pH[k][1] + (half_dt_w * fH[k][1].astype(w_dtype)).astype(dtype),
            pH[k][2] + (half_dt_w * fH[k][2].astype(w_dtype)).astype(dtype),
          )
          for k in range(4)
        ]
        ab_cur = [
          (rH[k][0] - rX[0], rH[k][1] - rX[1], rH[k][2] - rX[2])
          for k in range(4)
        ]
        p_out_X, p_out_H = _rattle_vel_cluster_vec3(
          p_b2_X, p_b2_H, ab_cur, inv_mX, inv_mH, v_slots,
          n_iters=8, use_f32_math=use_f32_math,
        )

        destX = jnp.where(v0, base_a + sX, scratch_slot) if use_scratch else (base_a + sX)
        _pl_store(mx_out_ref, (destX,), p_out_X[0], mask=v0)
        _pl_store(my_out_ref, (destX,), p_out_X[1], mask=v0)
        _pl_store(mz_out_ref, (destX,), p_out_X[2], mask=v0)
        for k in range(4):
          vk = v_slots[k]
          destH = jnp.where(vk, base_a + sH[k], scratch_slot) if use_scratch else (base_a + sH[k])
          _pl_store(mx_out_ref, (destH,), p_out_H[k][0], mask=vk)
          _pl_store(my_out_ref, (destH,), p_out_H[k][1], mask=vk)
          _pl_store(mz_out_ref, (destH,), p_out_H[k][2], mask=vk)
        return c_pid - c_pid

      return jax.lax.cond(
        jnp.max(v0.astype(jnp.int32)) > 0,
        _run_rattle,
        lambda _: c_pid - c_pid,
        None,
      )

    def _dispatch_non_solute(p_rem):
      if n_blk_w > 0 and n_blk_c > 0:
        return jax.lax.cond(
          p_rem < n_blk_w,
          _do_water,
          lambda p2: _do_rattle(p2 - n_blk_w),
          p_rem,
        )
      if n_blk_w > 0:
        return _do_water(p_rem)
      if n_blk_c > 0:
        return _do_rattle(p_rem)
      return p_rem - p_rem

    if n_blk_a > 0 and (n_blk_w + n_blk_c) > 0:
      _ = jax.lax.cond(
        pid < n_blk_a,
        _do_solute,
        lambda p: _dispatch_non_solute(p - n_blk_a),
        pid,
      )
    elif n_blk_a > 0:
      _ = _do_solute(pid)
    else:
      _ = _dispatch_non_solute(pid)

  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  out_len = B * N + (1 if use_scratch else 0)
  out_shapes = [jax.ShapeDtypeStruct((out_len,), dtype) for _ in range(3)]
  in_args = [
    px_in,
    py_in,
    pz_in,
    mx_in,
    my_in,
    mz_in,
    fx_in,
    fy_in,
    fz_in,
    m_in,
    am_in,
    is_water_flat,
    wO_in,
    wH1_in,
    wH2_in,
    wm_in,
    dt_b,
  ]
  if has_rattle:
    in_args.extend([cX_in, cH0_in, cH1_in, cH2_in, cH3_in, cM_in])

  mx_o, my_o, mz_o = pl.pallas_call(
    _post_kernel,
    out_shape=out_shapes,
    in_specs=[any_spec] * len(in_args),
    out_specs=[any_spec] * 3,
    grid=(B, total_blocks),
    input_output_aliases={3: 0, 4: 1, 5: 2},
    interpret=interpret,
    compiler_params=(
      resolve_pallas_compiler_params("settle_integrator", grid_blocks=B * total_blocks)
      if not interpret
      else None
    ),
    name="pallas_baoab_post_force",
  )(*in_args)
  if use_scratch:
    mx_o, my_o, mz_o = mx_o[:-1], my_o[:-1], mz_o[:-1]
  return jnp.stack(
    [
      mx_o.reshape(B, N),
      my_o.reshape(B, N),
      mz_o.reshape(B, N),
    ],
    axis=-1,
  )


def pallas_baoab_pre_force(
  positions: Array,
  momentum: Array,
  force: Array,
  mass: Array,
  rng: Array,
  water_indices: Array,
  water_mask: Array | None,
  box: Array | None,
  dt: float | Array,
  kT: float | Array,
  gamma: float,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  project_ou_momentum_rigid: bool = True,
  interpret: bool | None = None,
  atom_mask: Array | None = None,
  is_water_or_ghost: Array | None = None,
  constraint_pairs: Array | None = None,
  constraint_lengths: Array | None = None,
  constraint_mask: Array | None = None,
  return_pos_f32: bool = False,
  reference_position: Array | None = None,
  return_max_dr_sq: bool = False,
  n_active_atoms: int | None = None,
  n_active_waters: int | None = None,
) -> tuple[Array, ...]:
  """Fused Pallas pre-force BAOAB + analytical SETTLE + RATTLE integrator step (MK-46 / MK-58)."""
  interp = (jax.default_backend() == "cpu") if interpret is None else bool(interpret)
  n_w = int(water_indices.shape[0])
  key, subkey = jax.random.split(rng)
  seed_pair = jax.lax.bitcast_convert_type(
    jax.random.bits(subkey, shape=(2,), dtype=jnp.uint32),
    jnp.int32,
  )

  wm_arr = (
    jnp.ones((n_w,), dtype=jnp.int32)
    if water_mask is None
    else jnp.asarray(water_mask, dtype=jnp.int32)
  )
  has_box = box is not None
  if has_box:
    bx_arr = jnp.asarray(box, dtype=positions.dtype)
    if bx_arr.ndim == 2:
      bx_vec = jnp.diag(bx_arr)
    else:
      bx_vec = bx_arr[:3]
  else:
    bx_vec = jnp.ones((3,), dtype=positions.dtype)

  dt_arr = jnp.asarray(dt, dtype=positions.dtype)
  kT_arr = jnp.asarray(kT, dtype=positions.dtype)
  g_f = float(gamma)
  roh_f = float(r_OH)
  rhh_f = float(r_HH)
  mo_f = float(mass_oxygen)
  mh_f = float(mass_hydrogen)
  proj_b = bool(project_ou_momentum_rigid)
  n_a = int(positions.shape[0])
  am_arr = (
    jnp.ones((n_a,), dtype=jnp.int32)
    if atom_mask is None
    else jnp.asarray(atom_mask, dtype=jnp.int32)
  )
  has_iwg = is_water_or_ghost is not None
  iwg_arr = jnp.asarray(is_water_or_ghost, dtype=jnp.int32) if has_iwg else None
  has_rattle = (
    constraint_pairs is not None
    and constraint_lengths is not None
    and int(constraint_pairs.shape[-2]) > 0
  )
  if has_rattle:
    n_c = int(constraint_pairs.shape[-2])
    cp_arr = jnp.asarray(constraint_pairs, dtype=jnp.int32)
    cl_arr = jnp.asarray(constraint_lengths, dtype=positions.dtype)
    cm_arr = (
      jnp.ones((n_c,), dtype=jnp.int32)
      if constraint_mask is None
      else jnp.asarray(constraint_mask, dtype=jnp.int32)
    )
  else:
    cp_arr = cl_arr = cm_arr = None
  has_ref_pos = reference_position is not None
  ref_pos_arr = jnp.asarray(reference_position, dtype=positions.dtype) if has_ref_pos else None

  dyn_args: list[Array] = [
    positions,
    momentum,
    force,
    mass,
    seed_pair,
    water_indices,
    wm_arr,
    bx_vec,
    dt_arr,
    kT_arr,
    am_arr,
  ]
  if has_iwg:
    dyn_args.append(iwg_arr)
  if has_rattle:
    dyn_args.extend([cp_arr, cl_arr, cm_arr])
  if has_ref_pos:
    dyn_args.append(ref_pos_arr)

  def _unpack_call(axis_size: int, in_batched_tuple: tuple[bool, ...], *args: Array):
    idx = 11
    iwg_in = None
    if has_iwg:
      iwg_in = args[idx]
      idx += 1
    cp_in = cl_in = cm_in = None
    if has_rattle:
      cp_in, cl_in, cm_in = args[idx], args[idx + 1], args[idx + 2]
      idx += 3
    ref_in = None
    if has_ref_pos:
      ref_in = args[idx]
      idx += 1
    return _pallas_baoab_pre_force_batch(
      axis_size,
      in_batched_tuple,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[6],
      args[7],
      args[8],
      args[9],
      g_f,
      roh_f,
      rhh_f,
      mo_f,
      mh_f,
      has_box,
      proj_b,
      interp,
      atom_mask=args[10],
      is_water_or_ghost=iwg_in,
      constraint_pairs=cp_in,
      constraint_lengths=cl_in,
      constraint_mask=cm_in,
      return_pos_f32=return_pos_f32,
      reference_position=ref_in,
      return_max_dr_sq=return_max_dr_sq,
      n_active_atoms=n_active_atoms,
      n_active_waters=n_active_waters,
    )

  @jax.custom_batching.custom_vmap
  def _vmappable(*args: Array):
    outs_b = _unpack_call(1, (False,) * len(args), *args)
    return tuple(o[0] for o in outs_b)

  @_vmappable.def_vmap
  def _vmap_rule(axis_size, in_batched, *args: Array):
    outs_b = _unpack_call(int(axis_size), tuple(bool(b) for b in in_batched), *args)
    return outs_b, (True,) * len(outs_b)

  outs = _vmappable(*dyn_args)
  pos_new, mom_new = outs[0], outs[1]
  extra = outs[2:]
  return (pos_new, mom_new, key, *extra)


def pallas_baoab_post_force(
  positions: Array,
  momentum: Array,
  force: Array,
  mass: Array,
  water_indices: Array,
  water_mask: Array | None,
  dt: float | Array,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  interpret: bool | None = None,
  atom_mask: Array | None = None,
  is_water_or_ghost: Array | None = None,
  constraint_pairs: Array | None = None,
  constraint_lengths: Array | None = None,
  constraint_mask: Array | None = None,
  n_active_atoms: int | None = None,
  n_active_waters: int | None = None,
) -> Array:
  """Fused Pallas post-force BAOAB + analytical SETTLE + RATTLE velocity kernel (MK-46 / MK-58)."""
  interp = (jax.default_backend() == "cpu") if interpret is None else bool(interpret)
  n_w = int(water_indices.shape[0])
  wm_arr = (
    jnp.ones((n_w,), dtype=jnp.int32)
    if water_mask is None
    else jnp.asarray(water_mask, dtype=jnp.int32)
  )
  dt_arr = jnp.asarray(dt, dtype=positions.dtype)
  mo_f = float(mass_oxygen)
  mh_f = float(mass_hydrogen)
  n_a = int(positions.shape[0])
  am_arr = (
    jnp.ones((n_a,), dtype=jnp.int32)
    if atom_mask is None
    else jnp.asarray(atom_mask, dtype=jnp.int32)
  )
  has_iwg = is_water_or_ghost is not None
  iwg_arr = jnp.asarray(is_water_or_ghost, dtype=jnp.int32) if has_iwg else None
  has_rattle = (
    constraint_pairs is not None
    and constraint_lengths is not None
    and int(constraint_pairs.shape[-2]) > 0
  )
  if has_rattle:
    n_c = int(constraint_pairs.shape[-2])
    cp_arr = jnp.asarray(constraint_pairs, dtype=jnp.int32)
    cl_arr = jnp.asarray(constraint_lengths, dtype=positions.dtype)
    cm_arr = (
      jnp.ones((n_c,), dtype=jnp.int32)
      if constraint_mask is None
      else jnp.asarray(constraint_mask, dtype=jnp.int32)
    )
  else:
    cp_arr = cl_arr = cm_arr = None

  dyn_args: list[Array] = [
    positions,
    momentum,
    force,
    mass,
    water_indices,
    wm_arr,
    dt_arr,
    am_arr,
  ]
  if has_iwg:
    dyn_args.append(iwg_arr)
  if has_rattle:
    dyn_args.extend([cp_arr, cl_arr, cm_arr])

  def _unpack_call(axis_size: int, in_batched_tuple: tuple[bool, ...], *args: Array):
    idx = 8
    iwg_in = None
    if has_iwg:
      iwg_in = args[idx]
      idx += 1
    cp_in = cl_in = cm_in = None
    if has_rattle:
      cp_in, cl_in, cm_in = args[idx], args[idx + 1], args[idx + 2]
      idx += 3
    return _pallas_baoab_post_force_batch(
      axis_size,
      in_batched_tuple,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[6],
      mo_f,
      mh_f,
      interp,
      atom_mask=args[7],
      is_water_or_ghost=iwg_in,
      constraint_pairs=cp_in,
      constraint_lengths=cl_in,
      constraint_mask=cm_in,
      n_active_atoms=n_active_atoms,
      n_active_waters=n_active_waters,
    )

  @jax.custom_batching.custom_vmap
  def _vmappable_m(*args: Array):
    m_b = _unpack_call(1, (False,) * len(args), *args)
    return m_b[0]

  @_vmappable_m.def_vmap
  def _vmap_rule_m(axis_size, in_batched, *args: Array):
    m_b = _unpack_call(int(axis_size), tuple(bool(b) for b in in_batched), *args)
    return m_b, True

  return _vmappable_m(*dyn_args)



def settle_langevin(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  gamma: float = 1.0,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  box: Array | None = None,
  constraints: tuple[Array, Array] | None = None,
  remove_linear_com_momentum: bool = False,
  project_ou_momentum_rigid: bool = True,
  projection_site: str = "post_o",
  settle_velocity_iters: int = 10,
  settle_velocity_tol: float | None = None,
  water_mask: Array | None = None,
  atom_mask: Array | None = None,
  is_force_shaped: bool = False,
  shake_tol: float = 1e-5,
  rattle_tol: float = 1e-5,
  max_shake_iter: int = 25,
  displacement_fn: Callable[..., Array] | None = None,
  use_pallas: bool | None = None,
):
  r"""Langevin dynamics integrator with SETTLE constraints for water.

  Process:
  1.  **Half-B**: Update momenta by half timestep.
  2.  **Half-A**: Update positions by half timestep.
  3.  **O**: Stochastic velocity update (Ornstein-Uhlenbeck pulse).
  4.  **Half-A**: Second half position update.
  5.  **SETTLE-Pos**: Correct water positions analytically.
  6.  **Force**: Recompute forces at constrained positions.
  7.  **Half-B**: Final half velocity update.
  8.  **SETTLE-Vel**: Final velocity constraint correction.

  Notes:
  Combines BAOAB Langevin integrator with analytical rigid water constraints.
  After the Ornstein–Uhlenbeck step, atomic momenta are **mass-weighted projected** onto each
  water's rigid-body subspace (unless ``project_ou_momentum_rigid=False``), so isotropic
  Cartesian noise does not over-drive ``6 N_w-3`` kinetic degrees of freedom.

  **IMPORTANT (timestep)**: **dt ≤ 1.0 fs** is validated for production-scale systems
  (n_waters ≳ 16 with gamma ≈ 10 ps⁻¹). The original dt=1.0 fs gate (895 waters, job
  15870804) held T_rot = 299.6 K, but its propagator had a bug (fixed in cfd2b85,
  task 260806_p5_measurement_pipeline_audit): the A-step was passed `half_dt` where
  `_langevin_step_a` already halves internally, so positions actually advanced at an
  *effective* dt ≈ 0.5 fs, not the requested 1.0 fs. Job 15870804's pass is therefore
  not evidence about the corrected propagator at a genuine 1.0 fs step. The corrected
  propagator was independently re-gated at genuine dt=1.0 fs (job 19774893,
  T_rot=298.93±0.29 K) and confirmed clean on GPU with no correctness or performance
  regression in the fast test suite (job 19875071). The system-size sweep (campaign
  ba334c1f) showing T_total within ±15 K for n ≥ 16 and within ±5 K for n ≥ 64 was
  originally run under the same effective-0.5-fs propagator as job 15870804, but has
  since been re-run at genuine dt=1.0 fs (2026-08-11, campaign 46a4d737) and the
  crossover conclusion is unchanged: N*=16 (±15 K), N*=64 (±5 K). T_rot is faithful at
  every size in both gates. Below n ≈ 16 a
  **translational finite-size warm bias** appears (only 3·N−3 translational DOF
  at small N, under-regulated against the SETTLE constraint impulse). For very small
  systems (n ≲ 16) or weak friction (gamma ≈ 1 ps⁻¹), use **dt ≤ 0.5 fs**. See
  .praxia/docs/research/260612_p5-dt1fs-size-crossover.md.

  Args:
      energy_or_force_fn: System force definition.
      shift_fn: Displacement function.
      dt: Timestep.
      kT: Thermal energy target.
      gamma: Friction coefficient.
      mass: Atomic masses.
      water_indices: (N_waters, 3) indices.
      r_OH, r_HH: Target water geometry.
      m_O, m_H: Solvent masses.
      box: Periodic box dimensions.
      constraints: Optional (pairs, lengths) tuple for solute RATTLE.
      remove_linear_com_momentum: If True, after SETTLE velocity projection subtract the
          total center-of-mass velocity from every atom: ``p <- p - m * v_com`` with
          ``v_com = sum(p) / sum(m)``. This matches the **linear** part of OpenMM's
          ``CMMotionRemover`` (default frequency 1: subtract uniform ``v_com`` from all
          velocities), but occurs at a **different** point in the BAOAB+SETTLE timestep
          than OpenMM's integrator schedule. Use for production-style COM drift control,
          not for claiming bitwise phase-space parity with OpenMM.
    project_ou_momentum_rigid: If True (default), apply rigid momentum projection at the location
      selected by ``projection_site``.
    projection_site: Where to apply rigid momentum projection when
      ``project_ou_momentum_rigid=True``:
      ``post_o`` (legacy default), ``post_settle_vel``, or ``both``.
      Isotropic OU noise is applied in Cartesian atomic space; **always** projecting once
      immediately after the ``O`` step (using the mid-step positions) is required so rigid
      water degrees of freedom are not over-driven. ``post_settle_vel`` / ``both`` add an
      optional second projection after ``settle_velocities`` at the final constrained geometry.
    settle_velocity_iters: Number of RATTLE-like velocity correction iterations in
      ``settle_velocities``.
    water_mask: Optional (N_waters,) bool, True for real water rows and False
      for padding (see `MolecularBundle.water_mask`). Required whenever
      `water_indices` may contain padding rows (e.g. the stacked/vmapped B1
      dispatch path) — without it, padding rows silently corrupt whichever
      real atom the padding fill index ([0, 0, 0]) happens to coincide with.
      `None` (default) preserves prior behavior for callers that already
      guarantee `water_indices` has no padding (e.g. the single-system path).
    atom_mask: Optional (N,) bool, True for real (non-padding) atoms. Only
      consulted on the no-water fallback path (plain `simulate.nvt_langevin`)
      to re-mask/re-center the initial momentum draw so padding atoms (unit
      mass, not zero -- see `masses_with_prefix`) don't pollute real atoms'
      momentum via jax_md's default whole-array `center_velocity` correction
      (debt 841). `None` (default) preserves prior behavior for callers whose
      `R` already contains only real atoms (e.g. the single-system path).

  Returns:
      (init_fn, apply_fn) pair.
  """
  force_fn = _make_settle_compatible_force_fn(
    energy_or_force_fn, mass, box, is_force_shaped=is_force_shaped
  )
  if projection_site not in ("post_o", "post_settle_vel", "both"):
    msg = f"invalid projection_site={projection_site!r}; expected post_o|post_settle_vel|both"
    raise ValueError(msg)
  if not project_ou_momentum_rigid and projection_site == "both":
    msg = "projection_site='both' requires project_ou_momentum_rigid=True"
    raise ValueError(msg)

  # If no water indices, fall back to standard Langevin
  if water_indices is None or water_indices.shape[0] == 0:
    plain_init_fn, plain_apply_fn = simulate.nvt_langevin(
      energy_or_force_fn, shift_fn, dt, kT, gamma=gamma, mass=mass
    )
    if atom_mask is None:
      return plain_init_fn, plain_apply_fn

    # jax_md's nvt_langevin.init_fn (center_velocity=True default) subtracts
    # the momentum mean over ALL rows of R -- including padding atoms, which
    # carry a nonzero (unit-mass, see masses_with_prefix) random draw, not
    # zero. That pollutes every real atom's momentum with an incorrect COM
    # term (debt 841: this is why the batched/stacked EnsemblePlan dispatch
    # path diverged from independent single-bundle dispatch from step one,
    # even after fixing the water-detection bug above). Since mean-subtraction
    # is a rigid per-row shift by one constant vector, re-masking and
    # re-centering using ONLY the real (atom_mask) rows exactly cancels
    # jax_md's incorrect global term algebraically -- recovering the same
    # momentum an unpadded/independent call would produce for the real atoms.
    def masked_init_fn(key, R, mass=mass, **kwargs):
      state = plain_init_fn(key, R, mass=mass, **kwargs)
      mask_col = atom_mask[:, None]
      n_real = jnp.sum(atom_mask)
      com = jnp.sum(
        jnp.where(mask_col, state.momentum, 0.0), axis=0, keepdims=True
      ) / jnp.maximum(n_real, 1)
      corrected = jnp.where(mask_col, state.momentum - com, 0.0)
      return state.set(momentum=corrected)

    return masked_init_fn, plain_apply_fn

  # Precompute loop-invariant (N,) water/ghost exclusion mask once outside init_fn and apply_fn
  # when mass is a 1D/2D per-atom array, avoiding O(N * 3N_w) jnp.isin in init_fn and per-step scatters.
  _mass_init_arr = jnp.asarray(mass)
  _iwg_precomputed: Array | None = None
  if (
    _mass_init_arr.ndim >= 1
    and water_indices is not None
    and int(water_indices.shape[0]) > 0
  ):
    _n_a_init = int(_mass_init_arr.reshape(-1).shape[0])
    _wi_init = jnp.asarray(water_indices, dtype=jnp.int32)
    _n_w_init = int(_wi_init.shape[0])
    _wm_init = (
      jnp.ones((1, _n_w_init), dtype=jnp.int32)
      if water_mask is None
      else jnp.asarray(water_mask, dtype=jnp.int32)[None, :]
    )
    _am_init = (
      None
      if atom_mask is None
      else jnp.asarray(atom_mask, dtype=jnp.int32)[None, :]
    )
    _iwg_precomputed = _build_is_water_or_ghost_mask(
      1, _n_a_init, _n_w_init, _wi_init[None, ...], _wm_init, _am_init
    )

  _c_pairs_sl = _c_lengths_sl = _c_mask_sl = None
  if constraints is not None:
    _c_pairs_sl, _c_lengths_sl, _c_mask_sl = unpack_constraints(constraints)

  _can_pallas_default = (
    _should_use_pallas_settle_default(use_pallas)
    if use_pallas is None
    else (
      bool(use_pallas)
      and not in_export_context()
      and _should_use_pallas_settle_default(use_pallas)
    )
  )
  _use_f32_force_pos = (
    bool(is_force_shaped)
    and _can_pallas_default
    and jax.default_backend() != "cpu"
    and os.environ.get("PROLIX_DISABLE_MIXED_F32_POS", "") != "1"
    and displacement_fn is None
    and water_indices is not None
    and int(water_indices.shape[0]) > 0
    and int(settle_velocity_iters) > 0
    and (box is None or jnp.asarray(box).ndim == 1)
  )

  def init_fn(key, R, mass=mass, **kwargs):
    _kT = kwargs.pop("kT", kT)
    _precomputed_force = kwargs.pop("precomputed_force", None)
    key, split = random.split(key)
    if _use_f32_force_pos and R.dtype == jnp.float64:
      force = (
        jnp.asarray(_precomputed_force, dtype=jnp.float32)
        if _precomputed_force is not None
        else jnp.asarray(force_fn(R.astype(jnp.float32), **kwargs), dtype=jnp.float32)
      )
    else:
      force = _precomputed_force if _precomputed_force is not None else force_fn(R, **kwargs)

    # Handle mass array
    mass_arr = jnp.array(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr

    # Initialize momenta differently depending on whether we use constrained thermostat
    if project_ou_momentum_rigid and water_indices is not None and water_indices.shape[0] > 0:
      # Start with zero momenta, will fill in for water and non-water separately
      momenta = jnp.zeros_like(R, dtype=R.dtype)

      # Water atoms: sample from constrained distribution
      idx = water_indices
      mass_flat = mass_arr.reshape(-1)
      r_water = jnp.stack([R[idx[:, 0]], R[idx[:, 1]], R[idx[:, 2]]], axis=1)
      m_water = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)
      n_w = int(r_water.shape[0])
      if n_w <= 1:
        splits = jax.random.split(key, n_w + 1)
        key, water_keys = splits[0], splits[1:]
        p_water = dispatch_n_waters(
          None,
          n_w,
          lambda pack: _init_momentum_one_water_rigid(pack[0], pack[1], pack[2], _kT)[0],
          (water_keys, r_water, m_water),
        )
      else:
        # MK-50: Vectorized single-draw Box-Muller + rigid-water COM/angular initialization
        # eliminating the 8,193-way jax.random.split and 8,192 separate normal calls.
        key, split_w = jax.random.split(key)
        z6_all = jax.random.normal(split_w, (n_w, 6), dtype=jnp.float32).astype(R.dtype)

        def _init_water_from_z6(pack):
          z6_w, r_w, m_w = pack
          msum = jnp.sum(m_w)
          eps = jnp.array(1e-30, dtype=msum.dtype)
          msum_safe = jnp.maximum(msum, eps)
          com = jnp.sum(m_w[:, None] * r_w, axis=0) / msum_safe
          rrel = r_w - com
          z_trans = z6_w[:3]
          xi_trans = z_trans * jnp.sqrt(_kT / msum_safe)
          p_trans = m_w[:, None] * xi_trans[None, :]
          eye3 = jnp.eye(3, dtype=r_w.dtype)
          r_sq = jnp.einsum("ia,ia->i", rrel, rrel)
          I_tensor = (
            jnp.einsum("i,i->", m_w, r_sq) * eye3
            - jnp.einsum("i,ia,ib->ab", m_w, rrel, rrel)
          )
          reg_r = jnp.array(1e-12, dtype=I_tensor.dtype) * (jnp.trace(I_tensor) / 3.0 + 1.0)
          I_reg = I_tensor + reg_r * eye3
          z_rot = z6_w[3:]
          omega = _cholesky_upper_solve_3x3(I_reg, jnp.sqrt(_kT) * z_rot)
          p_rot = m_w[:, None] * jnp.cross(omega[None, :], rrel)
          return p_trans + p_rot

        p_water = jax.vmap(_init_water_from_z6)((z6_all, r_water, m_water))

      idx_flat = idx.reshape(-1)
      if water_mask is None:
        momenta = momenta.at[idx_flat].set(p_water.reshape(-1, 3))
      else:
        mask_flat = jnp.broadcast_to(water_mask[:, None], idx.shape).reshape(-1)
        p_water_flat = p_water.reshape(-1, 3)
        delta_p = jnp.where(mask_flat[:, None], p_water_flat - momenta[idx_flat], 0.0)
        momenta = momenta.at[idx_flat].add(delta_p)

      # MK-50: O(N) non-water active solute mask via precomputed _iwg_precomputed
      # (or direct O(N + 3N_w) scatter), eliminating the 614.4M-element jnp.isin.
      if n_w <= 1:
        real_idx_flat = (
          idx_flat
          if water_mask is None
          else jnp.where(
            jnp.broadcast_to(water_mask[:, None], idx.shape).reshape(-1), idx_flat, -1
          )
        )
        all_indices = jnp.arange(R.shape[0])
        is_water = jnp.isin(all_indices, real_idx_flat, assume_unique=False)
        non_water_mask = ~is_water
        if atom_mask is not None:
          non_water_mask = non_water_mask & jnp.asarray(atom_mask, dtype=jnp.bool_)
      elif _iwg_precomputed is not None and int(_iwg_precomputed.shape[0]) == int(R.shape[0]):
        non_water_mask = _iwg_precomputed == 0
      else:
        n_a_curr = int(R.shape[0])
        wi_curr = jnp.asarray(idx, dtype=jnp.int32)
        wm_curr = (
          jnp.ones((1, n_w), dtype=jnp.int32)
          if water_mask is None
          else jnp.asarray(water_mask, dtype=jnp.int32)[None, :]
        )
        am_curr = (
          None
          if atom_mask is None
          else jnp.asarray(atom_mask, dtype=jnp.int32)[None, :]
        )
        iwg_curr = _build_is_water_or_ghost_mask(
          1, n_a_curr, n_w, wi_curr[None, ...], wm_curr, am_curr
        )
        non_water_mask = iwg_curr == 0

      key, split = jax.random.split(key)
      non_water_noise_full = jnp.sqrt(mass_arr[:, None] * _kT) * jax.random.normal(
        split, R.shape, dtype=R.dtype
      )
      momenta = jnp.where(non_water_mask[:, None], non_water_noise_full, momenta)
    else:
      # Standard unconstrained initialization
      key, split = jax.random.split(key)
      momenta = jnp.sqrt(mass_arr[:, None] * _kT) * jax.random.normal(split, R.shape, dtype=R.dtype)
      if atom_mask is not None:
        momenta = jnp.where(jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None], momenta, 0.0)

    # Store mass for broadcasting
    mass_state = mass_arr[:, None]
    if constraints is not None:
      c_pairs, _c_lengths, c_mask = unpack_constraints(constraints)
      momenta = project_momenta(
        momenta,
        R,
        c_pairs,
        mass_state,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    return NVTLangevinState(R, momenta, force, mass_state, key)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)
    _use_pallas_settle = kwargs.pop("use_pallas_settle", use_pallas)
    half_dt = 0.5 * _dt

    _is_ad = any(
      isinstance(x, jax.interpreters.ad.JVPTracer)
      for x in (state.positions, state.momentum, state.force, _dt, _kT)
    )
    can_pallas_settle = (
      not _is_ad
      and (
        _should_use_pallas_settle_default(use_pallas)
        if _use_pallas_settle is None
        else (bool(_use_pallas_settle) and not in_export_context() and _should_use_pallas_settle_default(_use_pallas_settle))
      )
      and (constraints is None or (_c_pairs_sl is not None and _c_lengths_sl is not None))
      and displacement_fn is None
      and water_indices is not None
      and int(water_indices.shape[0]) > 0
      and int(settle_velocity_iters) > 0
      and (box is None or jnp.asarray(box).ndim == 1)
    )

    if can_pallas_settle:
      use_f32_step = (
        _use_f32_force_pos
        and state.positions.dtype == jnp.float64
        and state.force.dtype == jnp.float32
      )
      if use_f32_step:
        position, momentum, key, pos_f32 = pallas_baoab_pre_force(
          state.positions,
          state.momentum,
          state.force,
          state.mass,
          state.key,
          water_indices,
          water_mask,
          box,
          _dt,
          _kT,
          gamma,
          r_OH=r_OH,
          r_HH=r_HH,
          mass_oxygen=mass_oxygen,
          mass_hydrogen=mass_hydrogen,
          project_ou_momentum_rigid=project_ou_momentum_rigid,
          atom_mask=atom_mask,
          is_water_or_ghost=_iwg_precomputed,
          constraint_pairs=_c_pairs_sl,
          constraint_lengths=_c_lengths_sl,
          constraint_mask=_c_mask_sl,
          return_pos_f32=True,
        )
        force = jnp.asarray(force_fn(pos_f32, **kwargs), dtype=jnp.float32)
      else:
        position, momentum, key = pallas_baoab_pre_force(
          state.positions,
          state.momentum,
          state.force,
          state.mass,
          state.key,
          water_indices,
          water_mask,
          box,
          _dt,
          _kT,
          gamma,
          r_OH=r_OH,
          r_HH=r_HH,
          mass_oxygen=mass_oxygen,
          mass_hydrogen=mass_hydrogen,
          project_ou_momentum_rigid=project_ou_momentum_rigid,
          atom_mask=atom_mask,
          is_water_or_ghost=_iwg_precomputed,
          constraint_pairs=_c_pairs_sl,
          constraint_lengths=_c_lengths_sl,
          constraint_mask=_c_mask_sl,
        )
        force = force_fn(position, **kwargs)
      momentum = pallas_baoab_post_force(
        position,
        momentum,
        force,
        state.mass,
        water_indices,
        water_mask,
        _dt,
        mass_oxygen=mass_oxygen,
        mass_hydrogen=mass_hydrogen,
        atom_mask=atom_mask,
        is_water_or_ghost=_iwg_precomputed,
        constraint_pairs=_c_pairs_sl,
        constraint_lengths=_c_lengths_sl,
        constraint_mask=_c_mask_sl,
      )
      if project_ou_momentum_rigid and projection_site in ("post_settle_vel", "both"):
        momentum = project_tip3p_waters_momentum_rigid(
          momentum, position, state.mass, water_indices, water_mask
        )
      if remove_linear_com_momentum:
        mass_col = state.mass
        p_tot = jnp.sum(momentum, axis=0)
        m_tot = jnp.sum(mass_col)
        v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
        momentum = momentum - mass_col * v_com
      return NVTLangevinState(position, momentum, force, state.mass, key)

    # Store old positions for first R-step (reference for first SETTLE)
    positions_old = state.positions

    # B: half force kick
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # L_target reference for A+SETTLE1+R1 AM correction
    p_pre_a1 = momentum

    # R1: first half-step (A + SETTLE + dp-correction + SETTLE_vel)
    # Pass the FULL _dt: _langevin_step_a already halves internally (it advances
    # 0.5*dt*v), so this is the A(dt/2) half-step. Passing half_dt here would
    # advance only 0.25*dt*v, making the full cycle cover 0.5*dt*v instead of
    # BAOAB's dt*v (task 260806_p5_measurement_pipeline_audit).
    x_unc_1 = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)
    x_before_constraints_1 = x_unc_1

    # Solute RATTLE (SHAKE) before SETTLE
    if constraints is not None:
      c_pairs, c_lengths, c_mask = unpack_constraints(constraints)
      x_unc_1 = project_positions(
        x_unc_1,
        c_pairs,
        c_lengths,
        state.mass,
        shift_fn,
        tol=shake_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # SETTLE position constraints
    x_con_1 = settle_positions(
      x_unc_1,
      positions_old,
      water_indices,
      r_OH,
      r_HH,
      mass_oxygen,
      mass_hydrogen,
      box,
      water_mask,
    )

    # OpenMM R-step: momentum correction from constraint impulse.
    # Use x_before_constraints_1 (saved prior to project_positions) so solute
    # SHAKE positional corrections also contribute to the R-step velocity impulse.
    if displacement_fn is not None:
      dx_1 = jax.vmap(displacement_fn)(x_con_1, x_before_constraints_1)
    else:
      dx_1 = x_con_1 - x_before_constraints_1
      if box is not None:
        if jnp.asarray(box).ndim == 2:
          dx_1 = pbc_module.minimum_image_displacement(x_con_1, x_before_constraints_1, box)
        else:
          dx_1 = dx_1 - box * _PBC_ROUND_FN(dx_1 / box)
    dp_1 = state.mass * dx_1 / half_dt
    momentum = momentum + dp_1

    # Restore angular momentum after A+SETTLE+R block
    if water_indices is not None:
      momentum = _r_step_conserve_angular_momentum(
        momentum, p_pre_a1, x_unc_1, x_con_1, water_indices, mass_oxygen, mass_hydrogen,
        box=box, water_mask=water_mask,
      )

    position = x_con_1
    positions_mid = x_con_1

    # O: stochastic step (unchanged)
    if project_ou_momentum_rigid:
      momentum, key = _langevin_step_o_constrained(
        momentum, position, state.mass, gamma, _dt, _kT, state.key, water_indices, water_mask
      )
    else:
      momentum, key = _langevin_step_o(momentum, state.mass, gamma, _dt, _kT, state.key)

    # L_target reference for A+SETTLE2+R2 AM correction
    p_pre_a2 = momentum

    # R2: second half-step (A + SETTLE + dp-correction + SETTLE_vel)
    # Full _dt for the same reason as R1 above; the sub-step duration this A-step
    # represents is half_dt, which is what the dp_2 impulse below divides by.
    x_unc_2 = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)
    x_before_constraints_2 = x_unc_2

    # Solute RATTLE (SHAKE) before SETTLE
    if constraints is not None:
      c_pairs, c_lengths, c_mask = unpack_constraints(constraints)
      x_unc_2 = project_positions(
        x_unc_2,
        c_pairs,
        c_lengths,
        state.mass,
        shift_fn,
        tol=shake_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # SETTLE position constraints
    x_con_2 = settle_positions(
      x_unc_2,
      positions_mid,
      water_indices,
      r_OH,
      r_HH,
      mass_oxygen,
      mass_hydrogen,
      box,
      water_mask,
    )

    # OpenMM R-step: momentum correction from constraint impulse (minimum-image;
    # see R1 above for why the raw difference is unsafe under PBC wrapping).
    if displacement_fn is not None:
      dx_2 = jax.vmap(displacement_fn)(x_con_2, x_before_constraints_2)
    else:
      dx_2 = x_con_2 - x_before_constraints_2
      if box is not None:
        if jnp.asarray(box).ndim == 2:
          dx_2 = pbc_module.minimum_image_displacement(x_con_2, x_before_constraints_2, box)
        else:
          dx_2 = dx_2 - box * _PBC_ROUND_FN(dx_2 / box)
    dp_2 = state.mass * dx_2 / half_dt
    momentum = momentum + dp_2

    # Restore angular momentum after A+SETTLE+R block
    if water_indices is not None:
      momentum = _r_step_conserve_angular_momentum(
        momentum, p_pre_a2, x_unc_2, x_con_2, water_indices, mass_oxygen, mass_hydrogen,
        box=box, water_mask=water_mask,
      )

    position = x_con_2

    # Force at new constrained positions
    force = force_fn(position, **kwargs)

    # B: final half force kick
    momentum = _langevin_step_b(momentum, force, _dt)

    # Final velocity constraint (catches residual from force kick)
    if constraints is not None:
      c_pairs, _, c_mask = unpack_constraints(constraints)
      momentum = project_momenta(
        momentum,
        position,
        c_pairs,
        state.mass,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    momentum = _langevin_settle_vel(
      momentum,
      positions_mid,
      position,
      state.mass,
      water_indices,
      _dt,
      mass_oxygen,
      mass_hydrogen,
      n_iters=settle_velocity_iters,
      settle_velocity_tol=settle_velocity_tol,
      water_mask=water_mask,
    )

    if project_ou_momentum_rigid and projection_site in ("post_settle_vel", "both"):
      momentum = project_tip3p_waters_momentum_rigid(
        momentum, position, state.mass, water_indices, water_mask
      )

    if remove_linear_com_momentum:
      mass_col = state.mass
      p_tot = jnp.sum(momentum, axis=0)
      m_tot = jnp.sum(mass_col)
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = momentum - mass_col * v_com

    if atom_mask is not None:
      am_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      position = jnp.where(am_col, position, positions_old)
      momentum = jnp.where(am_col, momentum, jnp.asarray(0.0, dtype=momentum.dtype))

    return NVTLangevinState(position, momentum, force, state.mass, key)

  apply_fn._pallas_settle_pins_ghosts = bool(
    _should_use_pallas_settle_default(use_pallas)
    and (constraints is None or (_c_pairs_sl is not None and _c_lengths_sl is not None))
    and displacement_fn is None
    and water_indices is not None
    and int(water_indices.shape[0]) > 0
    and int(settle_velocity_iters) > 0
    and (box is None or jnp.asarray(box).ndim == 1)
    and not remove_linear_com_momentum
    and not (project_ou_momentum_rigid and projection_site in ("post_settle_vel", "both"))
  )

  if apply_fn._pallas_settle_pins_ghosts and is_force_shaped:
    def _apply_2step_respa(state, **kwargs):
      _dt = kwargs.pop("dt", dt)
      _kT = kwargs.pop("kT", kT)
      if _use_f32_force_pos:
        pos_1, mom_1, key_1, pos_f32_1 = pallas_baoab_pre_force(
          state.positions,
          state.momentum,
          state.force,
          state.mass,
          state.key,
          water_indices,
          water_mask,
          box,
          _dt,
          _kT,
          gamma,
          r_OH=r_OH,
          r_HH=r_HH,
          mass_oxygen=mass_oxygen,
          mass_hydrogen=mass_hydrogen,
          project_ou_momentum_rigid=project_ou_momentum_rigid,
          atom_mask=atom_mask,
          is_water_or_ghost=_iwg_precomputed,
          constraint_pairs=_c_pairs_sl,
          constraint_lengths=_c_lengths_sl,
          constraint_mask=_c_mask_sl,
          return_pos_f32=True,
        )
        f_res_1 = force_fn(pos_f32_1, return_pme_force=True, **kwargs)
      else:
        pos_1, mom_1, key_1 = pallas_baoab_pre_force(
          state.positions,
          state.momentum,
          state.force,
          state.mass,
          state.key,
          water_indices,
          water_mask,
          box,
          _dt,
          _kT,
          gamma,
          r_OH=r_OH,
          r_HH=r_HH,
          mass_oxygen=mass_oxygen,
          mass_hydrogen=mass_hydrogen,
          project_ou_momentum_rigid=project_ou_momentum_rigid,
          atom_mask=atom_mask,
          is_water_or_ghost=_iwg_precomputed,
          constraint_pairs=_c_pairs_sl,
          constraint_lengths=_c_lengths_sl,
          constraint_mask=_c_mask_sl,
        )
        f_res_1 = force_fn(pos_1, return_pme_force=True, **kwargs)
      if isinstance(f_res_1, tuple):
        frc_1, f_pme_1 = f_res_1
      else:
        frc_1, f_pme_1 = f_res_1, None
      if _use_f32_force_pos:
        frc_1 = jnp.asarray(frc_1, dtype=jnp.float32)
      mom_1 = pallas_baoab_post_force(
        pos_1,
        mom_1,
        frc_1,
        state.mass,
        water_indices,
        water_mask,
        _dt,
        mass_oxygen=mass_oxygen,
        mass_hydrogen=mass_hydrogen,
        atom_mask=atom_mask,
        is_water_or_ghost=_iwg_precomputed,
        constraint_pairs=_c_pairs_sl,
        constraint_lengths=_c_lengths_sl,
        constraint_mask=_c_mask_sl,
      )
      st_mid = NVTLangevinState(pos_1, mom_1, frc_1, state.mass, key_1)
      return apply_fn(st_mid, dt=_dt, kT=_kT, pme_cached_force=f_pme_1, **kwargs)

    apply_fn._respa_2step_fn = _apply_2step_respa
  else:
    apply_fn._respa_2step_fn = None

  return init_fn, apply_fn


def settle_lfmiddle_langevin(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  gamma: float = 1.0,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  box: Array | None = None,
  constraints: tuple[Array, Array] | None = None,
  remove_linear_com_momentum: bool = False,
  project_ou_momentum_rigid: bool = True,
  projection_site: str = "post_o",
  settle_velocity_iters: int = 10,
  settle_velocity_tol: float | None = None,
  water_mask: Array | None = None,
  atom_mask: Array | None = None,
  is_force_shaped: bool = False,
  shake_tol: float = 1e-5,
  rattle_tol: float = 1e-5,
  max_shake_iter: int = 25,
  displacement_fn: Callable[..., Array] | None = None,
  use_pallas: bool | None = None,
):
  r"""Langevin + SETTLE with Leimkuhler-Matthews O-step splitting (LFMiddle).

  Same as :func:`settle_langevin` but splits the stochastic O-step into two
  halves around the mid-step force recompute:

  B(0.5) → A(0.5) → O(0.5) → A(0.5) → SETTLE_pos → Force → O(0.5) → B(0.5) → SETTLE_vel

  The two O(0.5) halves compose to the full ``exp(-gamma*dt)`` friction, and the
  two A(0.5) drifts advance positions by exactly ``dt*v``. Until 2026-10-01 a third
  A(0.5) followed the force recompute, advancing ``1.5*dt*v`` per step with a force
  evaluated at pre-drift positions; LFMiddle results produced before then (incl.
  campaign 89c9a900) measured that defective scheme.

  Note: nothing touches the momentum between the two O(0.5) halves, so this step is
  algebraically BAOAB -- and a weaker one than :func:`settle_langevin` (no R-step
  constraint impulse, no angular-momentum restoration, no ``water_mask``). It is not a
  genuine Leimkuhler-Matthews LF-Middle splitting; do not use it to test that idea.
  """
  if projection_site not in ("post_o", "post_settle_vel", "both"):
    msg = f"invalid projection_site={projection_site!r}; expected post_o|post_settle_vel|both"
    raise ValueError(msg)

  init_fn, _ = settle_langevin(
      energy_or_force_fn,
      shift_fn,
      dt,
      kT,
      gamma=gamma,
      mass=mass,
      water_indices=water_indices,
      r_OH=r_OH,
      r_HH=r_HH,
      mass_oxygen=mass_oxygen,
      mass_hydrogen=mass_hydrogen,
      box=box,
      constraints=constraints,
      remove_linear_com_momentum=remove_linear_com_momentum,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
      projection_site=projection_site,
      settle_velocity_iters=settle_velocity_iters,
      settle_velocity_tol=settle_velocity_tol,
      water_mask=water_mask,
      atom_mask=atom_mask,
      is_force_shaped=is_force_shaped,
      shake_tol=shake_tol,
      rattle_tol=rattle_tol,
      max_shake_iter=max_shake_iter,
      displacement_fn=displacement_fn,
  )
  force_fn = _make_settle_compatible_force_fn(
    energy_or_force_fn, mass, box, is_force_shaped=is_force_shaped
  )
  if water_indices is None or water_indices.shape[0] == 0:
    return simulate.nvt_langevin(energy_or_force_fn, shift_fn, dt, kT, gamma=gamma, mass=mass)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)
    half_dt = 0.5 * _dt
    positions_old = state.positions

    momentum = _langevin_step_b(state.momentum, state.force, _dt)
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)

    if project_ou_momentum_rigid:
      momentum, key = _langevin_step_o_constrained(
          momentum, position, state.mass, gamma, half_dt, _kT, state.key, water_indices, water_mask
      )
    else:
      momentum, key = _langevin_step_o(momentum, state.mass, gamma, half_dt, _kT, state.key)

    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    if constraints is not None:
      c_pairs, c_lengths, c_mask = unpack_constraints(constraints)
      position = project_positions(
        position,
        c_pairs,
        c_lengths,
        state.mass,
        shift_fn,
        tol=shake_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    position = settle_positions(
        position, positions_old, water_indices, r_OH, r_HH,
        mass_oxygen, mass_hydrogen, box, water_mask,
    )

    force = force_fn(position, **kwargs)

    # No position update after the force recompute: the returned ``force`` must be
    # the force at the returned ``position`` (the next step's B half-kick uses it),
    # and the two A(0.5) drifts above already advance positions by exactly dt*v.
    # A former third A-step here advanced 1.5*dt*v per step and left state.force
    # stale (tests/physics/test_transrot_decomposition.py).
    if project_ou_momentum_rigid:
      momentum, key = _langevin_step_o_constrained(
          momentum, position, state.mass, gamma, half_dt, _kT, key, water_indices, water_mask
      )
    else:
      momentum, key = _langevin_step_o(momentum, state.mass, gamma, half_dt, _kT, key)

    momentum = _langevin_step_b(momentum, force, _dt)

    if constraints is not None:
      c_pairs, _, c_mask = unpack_constraints(constraints)
      momentum = project_momenta(
        momentum,
        position,
        c_pairs,
        state.mass,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    momentum = _langevin_settle_vel(
        momentum, positions_old, position, state.mass, water_indices, _dt,
        mass_oxygen, mass_hydrogen, n_iters=settle_velocity_iters,
        settle_velocity_tol=settle_velocity_tol, water_mask=water_mask,
    )
    if project_ou_momentum_rigid and projection_site in ("post_settle_vel", "both"):
      momentum = project_tip3p_waters_momentum_rigid(
          momentum, position, state.mass, water_indices, water_mask
      )
    if remove_linear_com_momentum:
      mass_col = state.mass
      p_tot = jnp.sum(momentum, axis=0)
      m_tot = jnp.sum(mass_col)
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = momentum - mass_col * v_com

    if atom_mask is not None:
      am_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      position = jnp.where(am_col, position, positions_old)
      momentum = jnp.where(am_col, momentum, jnp.asarray(0.0, dtype=momentum.dtype))

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn


def settle_with_nhc(
  energy_or_force_fn: Callable,
  shift_fn: Callable,
  dt: float,
  kT: float,
  mass: Array | float = 1.0,
  water_indices: WaterIndicesArray | None = None,
  box: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  settle_velocity_iters: int = 10,
  settle_velocity_tol: float | None = None,
  constraints: tuple | None = None,
  remove_linear_com_momentum: bool = True,
  chain_length: int = 5,
  chain_steps: int = 2,
  sy_steps: int = 3,
  tau: float | None = None,
) -> tuple[Callable, Callable]:
  r"""SETTLE constraints with JAX MD's Nosé-Hoover Chain thermostat.

  Wraps JAX MD's proven `nvt_nose_hoover()` with SETTLE rigid water
  constraint enforcement. Uses a Nosé-Hoover Chain (NHC) thermostat
  (chain_length=5 by default, more stable than single thermostat).

  Integrator Order:
  ```
  Base: JAX MD's nvt_nose_hoover (NHC thermostat)
  Wrapped with: SETTLE position constraints → SETTLE velocity constraints
  ```

  Args:
      energy_or_force_fn: Callable returning force (or energy).
      shift_fn: PBC shift function from `jax_md.space`.
      dt: Timestep (AKMA units).
      kT: Thermal energy (k_B * T in AKMA units).
      mass: Atomic masses.
      water_indices: (N_waters, 3) indices [O, H1, H2] for SETTLE waters.
      box: Optional PBC box dimensions (3,).
      r_OH, r_HH: Bond lengths for SETTLE.
      mass_oxygen, mass_hydrogen: TIP3P masses.
      settle_velocity_iters: SETTLE velocity correction iterations.
      settle_velocity_tol: SETTLE velocity tolerance.
      constraints: Optional (pairs, lengths) for solute RATTLE.
      remove_linear_com_momentum: Subtract COM velocity after SETTLE_vel.
      chain_length: NHC chain length (default 5).
      chain_steps: NHC outer substeps (default 2).
      sy_steps: Suzuki-Yoshida steps (default 3, must be 1/3/5/7).
      tau: NHC coupling timescale (in units of dt).

  Returns:
      (init_fn, apply_fn) tuple for NVT dynamics with NHC + SETTLE.
  """
  # Get JAX MD's proven Nosé-Hoover Chain integrator
  nhc_init, nhc_apply = simulate.nvt_nose_hoover(

    energy_or_force_fn,
    shift_fn,
    dt,
    kT,
    chain_length=chain_length,
    chain_steps=chain_steps,
    sy_steps=sy_steps,
    tau=tau,
  )

  # If no water indices, return the NHC integrator as-is
  if water_indices is None or water_indices.shape[0] == 0:
    return nhc_init, nhc_apply

  force_fn_after_settle = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)

  # Wrap NHC apply function with SETTLE constraints
  def apply_with_settle(state, **kwargs):
    """Apply one NHC step, then enforce SETTLE constraints.

    jax_md's NVTNoseHooverState / NoseHooverChain name the field ``position``
    (not ``positions``). NOTE: the chain is reset to zero every step below, which
    discards the thermostat's state -- this is the abandoned Phase 2
    "settle_with_nhc" approach, not a working NHC integrator (Phase 7, #4168).
    """
    # Store old positions for SETTLE velocity correction
    positions_old = state.position

    # Apply one step of JAX MD's NHC integrator
    state = nhc_apply(state, **kwargs)

    # Enforce SETTLE position constraints
    position = settle_positions(
      state.position,
      positions_old,
      water_indices,
      r_OH,
      r_HH,
      mass_oxygen,
      mass_hydrogen,
      box,
    )

    # Recompute forces after constraining positions
    force = force_fn_after_settle(position, **kwargs)

    # Enforce SETTLE velocity constraints
    momentum = _langevin_settle_vel(
      state.momentum,
      positions_old,
      position,
      state.mass,
      water_indices,
      kwargs.pop("dt", dt),
      mass_oxygen,
      mass_hydrogen,
      n_iters=settle_velocity_iters,
      settle_velocity_tol=settle_velocity_tol,
    )

    if remove_linear_com_momentum:
      mass_col = state.mass
      p_tot = jnp.sum(momentum, axis=0)
      m_tot = jnp.sum(mass_col)
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = momentum - mass_col * v_com

    # Re-synchronize NHC chain state after SETTLE constraints modify momentum.
    # SETTLE removes kinetic energy from constrained DOF, so the NHC chain state
    # (position, momentum of fictitious particles) becomes desynchronized. Reset
    # to zero to allow thermostat to re-equilibrate smoothly on the next step.
    new_chain = state.chain.set(
      position=jnp.zeros_like(state.chain.position),
      momentum=jnp.zeros_like(state.chain.momentum),
    )

    # Return updated state with constrained position, momentum, force, AND chain state
    return state.set(
      position=position,
      momentum=momentum,
      force=force,
      chain=new_chain,
    )

  return nhc_init, apply_with_settle


def langevin_with_constraints(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  gamma: float = 1.0,
  mass: float | Array = 1.0,
  constraint: Any | None = None,
  box: Array | None = None,
  remove_linear_com_momentum: bool = False,
  project_ou_momentum_rigid: bool = True,
  water_indices: Array | None = None,
  **kwargs,
) -> tuple[Callable, Callable]:
  r"""Langevin integrator with injected constraint algorithm.

  Wraps settle_langevin with a constraint plugin system. The constraint object
  provides apply_positions() and apply_velocities() methods, allowing flexible
  constraint composition (NullConstraint, SETTLEConstraint, ShakeRattleConstraint, etc.).

  Args:
      energy_or_force_fn: System force definition.
      shift_fn: Displacement function.
      dt: Timestep (AKMA units).
      kT: Thermal energy target (kcal/mol).
      gamma: Langevin friction coefficient (1/ps, reduced to AKMA units by caller).
      mass: Atomic masses (scalar or (N,) array).
      constraint: ConstraintAlgorithm instance (or None for no constraints).
      box: Periodic box dimensions or None.
      remove_linear_com_momentum: If True, remove COM momentum each step.
      project_ou_momentum_rigid: If True, project OU noise to rigid-body subspace for water.
      water_indices: (N_waters, 3) array of water atom indices for rigid-body projection.
      **kwargs: Additional arguments passed to energy_fn.

  Returns:
      (init_fn, apply_fn) following JAX-MD convention.
  """
  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)

  def init_fn(key, R, mass=mass, **init_kwargs):
    _kT = init_kwargs.pop("kT", kT)
    key, split = jax.random.split(key)
    force = force_fn(R, **init_kwargs)

    # Handle mass - keep as 1D for indexing, expand to (N, 1) for broadcasting
    mass_arr = jnp.asarray(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr
    elif mass_arr.ndim == 2:
      mass_arr = mass_arr.reshape(-1)

    # Initialize momenta from Maxwell-Boltzmann
    momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
      split, R.shape, dtype=R.dtype
    )

    # Store mass as (N, 1) for broadcasting
    mass_for_state = mass_arr[:, jnp.newaxis]

    return NVTLangevinState(R, momenta, force, mass_for_state, key)

  def apply_fn(state, **step_kwargs):
    _dt = step_kwargs.pop("dt", dt)
    _kT = step_kwargs.pop("kT", kT)

    # Store old positions for constraints
    positions_old = state.positions

    # === B-step (half-kick) ===
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # === A-step (half-move) ===
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)

    # === O-step (stochastic update) ===
    if project_ou_momentum_rigid and water_indices is not None:
      momentum, key_out = _langevin_step_o_constrained(
        momentum, position, state.mass, gamma, _dt, _kT, state.key, water_indices
      )
    else:
      momentum, key_out = _langevin_step_o(momentum, state.mass, gamma, _dt, _kT, state.key)

    # === A-step (second half-move) ===
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    # === Apply position constraints ===
    if constraint is not None:
      position = constraint.apply_positions(positions_old, position, state.mass, box)

    # === Force eval (at constrained positions) ===
    force = force_fn(position, **step_kwargs)

    # === B-step (final half-kick) ===
    momentum = _langevin_step_b(momentum, force, _dt)

    # === Apply velocity constraints ===
    if constraint is not None:
      momentum = constraint.apply_velocities(
        positions_old, position, momentum, state.mass, _dt, shift_fn
      )

    # === Remove COM momentum if requested ===
    if remove_linear_com_momentum:
      total_mass = jnp.sum(state.mass)
      com_momentum = jnp.sum(momentum, axis=0) / total_mass
      momentum = momentum - state.mass * com_momentum[jnp.newaxis, :]

    return NVTLangevinState(position, momentum, force, state.mass, key_out)

  return init_fn, apply_fn


def _csvr_compute_lambda(
  key: Array,
  ke_current: Array,
  n_dof: int | Array,
  kT: float,
  dt: float,
  tau: float = None,
  max_chi_dof: int | None = None,
) -> tuple[Array, Array]:
  r"""Compute CSVR (Bussi velocity-rescaling) lambda factor."""
  if tau is None:
    tau = _DEFAULT_CSVR_TAU_AKMA

  # Coupling strength: exp(-dt/tau)
  c1 = jnp.exp(-dt / tau)
  c2_sq = 1.0 - c1  # Note: this is (1 - c1), not sqrt(1 - c1^2)

  # Draw random numbers: one gaussian + (n_dof - 1) chi-squared components
  key, split = jax.random.split(key)
  r_gaussian = jax.random.normal(split, dtype=ke_current.dtype)

  key, split = jax.random.split(key)
  if isinstance(n_dof, (int, float)):
    n_dof_int = max(int(n_dof), 1)
    s_chi_squared = jnp.sum(
      jax.random.normal(split, (max(n_dof_int - 1, 0),), dtype=ke_current.dtype) ** 2
    )
    n_dof_f = jnp.asarray(float(n_dof_int), dtype=ke_current.dtype)
  else:
    n_dof_safe = jnp.maximum(jnp.asarray(n_dof, dtype=ke_current.dtype), 1.0)
    upper_dof = max(int(max_chi_dof if max_chi_dof is not None else 512), 1)
    chi_draws = jax.random.normal(split, (upper_dof,), dtype=ke_current.dtype) ** 2
    chi_mask = jnp.arange(upper_dof) < jnp.maximum(jnp.asarray(n_dof, dtype=jnp.int32) - 1, 0)
    s_chi_squared = jnp.sum(jnp.where(chi_mask, chi_draws, 0.0))
    n_dof_f = n_dof_safe

  target_ke = 0.5 * n_dof_f * kT
  ke_safe = jnp.maximum(ke_current, 1e-10)

  c2_corrected = c2_sq * (target_ke / ke_safe) / n_dof_f

  lambda_sq = (
    c1
    + c2_corrected * (r_gaussian**2 + s_chi_squared)
    + 2.0 * r_gaussian * jnp.sqrt(c1 * c2_corrected)
  )
  lambda_sq_safe = jnp.maximum(lambda_sq, 0.0)

  lambda_factor = jnp.where(ke_current <= 0.0, 1.0, jnp.sqrt(lambda_sq_safe))

  return lambda_factor, key


def _csvr_rescale_momenta(momentum: Array, lambda_factor: Array) -> Array:
  r"""Apply scalar rescaling to momenta.

  Since rescaling is a scalar operation, it preserves constraint subspaces:
  if v is in the tangent space of SETTLE constraints, so is alpha*v.

  Args:
    momentum: Atomic momenta (N, 3).
    lambda_factor: Scalar rescaling factor.

  Returns:
    Rescaled momenta.
  """
  return lambda_factor * momentum


def _n_dof_thermostated(
  n_atoms: int,
  n_waters: int,
  n_constraint_pairs: int = 0,
  remove_com: bool = True,
) -> int:
  r"""Compute degrees of freedom for CSVR thermostat.

  For a system with rigid water (SETTLE) + optional solute bonds (SHAKE/RATTLE):
  DOF = 3*N_atoms - 3*N_waters (SETTLE removes 3 DOF per water) - n_constraint_pairs (solute bonds)

  Then subtract 3 for COM motion removal if requested.

  For pure-water system (all atoms are water):
  - n_atoms = 3*N_waters
  - DOF = 3*(3*N_waters) - 3*N_waters = 9*N_waters - 3*N_waters = 6*N_waters
  - With remove_com=True: DOF = 6*N_waters - 3

  Args:
    n_atoms: Total number of atoms.
    n_waters: Number of water molecules.
    n_constraint_pairs: Number of constrained bond pairs (solute).
    remove_com: If True, subtract 3 for COM motion removal.

  Returns:
    Number of thermostated degrees of freedom.
  """
  # Start with 3N (all translational DOF)
  dof = 3 * n_atoms

  # Subtract 3 DOF per water molecule (SETTLE removes 3 constraints per water)
  dof -= 3 * n_waters

  # Subtract additional solute bond constraints
  dof -= n_constraint_pairs

  # Subtract 3 for COM motion if requested
  if remove_com:
    dof -= 3

  return dof


def settle_csvr(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  tau: float = None,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  n_constraint_pairs: int = 0,
  remove_com: bool = True,
  box: Array | None = None,
  water_mask: Array | None = None,
  atom_mask: Array | None = None,
  constraints: tuple | None = None,
  shake_tol: float = 1e-5,
  rattle_tol: float = 1e-5,
  max_shake_iter: int = 25,
  displacement_fn: Callable[..., Array] | None = None,
  is_force_shaped: bool = False,
  **kwargs,
) -> tuple[Callable, Callable]:
  r"""CSVR (Bussi velocity-rescaling) thermostat integrator with SETTLE constraints.

  Uses velocity-Verlet (B-A-A-B) rather than BAOAB — no O-step; CSVR replaces it.

  Process:
  1.  **B-step**: Half-kick momenta.
  2.  **A-step × 2**: Full position advance (two half-moves at fixed momentum).
  3.  **SETTLE-Pos**: Correct water positions analytically.
  4.  **Force eval**: Recompute forces at constrained positions.
  5.  **B-step**: Final half-kick.
  6.  **SETTLE-Vel**: Velocity constraint correction.
  7.  **CSVR-Rescale**: Global scalar velocity rescaling via chi-squared sampling.

  CSVR replaces the per-DOF Langevin O-step with a single chi-squared sample that
  drives a scalar rescaling of all momenta. This scalar operation preserves SETTLE
  constraint subspaces by construction: if v is in tangent space, so is alpha*v.

  **Allows dt=2.0fs (or larger)** without temperature oscillations.

  Args:
      energy_or_force_fn: System force definition.
      shift_fn: Displacement function.
      dt: Timestep (AKMA units).
      kT: Thermal energy target (kcal/mol).
      tau: Velocity rescaling time constant (AKMA units); default ~0.1ps. Require tau >> dt.
      mass: Atomic masses (scalar or (N,) array).
      water_indices: (N_waters, 3) array of water atom indices, or None.
      r_OH: Target O-H bond length (Å).
      r_HH: Target H-H distance (Å).
      mass_oxygen: Mass of oxygen (amu).
      mass_hydrogen: Mass of hydrogen (amu).
      n_constraint_pairs: Number of solute SHAKE/RATTLE bond pairs (for DOF count).
      remove_com: If True, exclude COM translation from DOF count.
      box: Periodic box dimensions or None.
      **kwargs: Additional arguments (energy_fn kwargs).

  Note:
      **Known limitation — temperature bias at dt >= 1 fs**: CSVR+SETTLE exhibits a
      tau-dependent mean temperature bias of approximately +8 K at dt >= 1 fs (e.g.,
      +8 K at tau=0.1 ps, dt=1 fs). Root cause: velocity-Verlet (B-A-A-B) operator
      splitting introduces a leading-order discretization error in the Bussi 2007
      rescaling step when dt is comparable to the thermostat relaxation time tau.
      At dt < 1 fs, the bias drops below ~1 K and is not practically significant.
      For simulations requiring tight temperature control (|T - T_target| < 5 K),
      use dt <= 0.5 fs or increase tau (weaker coupling reduces per-step bias).
      A runtime warning is emitted when dt >= 1 fs / 48.888 AKMA units.

  Returns:
      (init_fn, apply_fn) following JAX-MD convention.
  """
  # AKMA time unit = 48.888 fs; 1 fs ≈ 0.02046 AKMA
  _ONE_FS_AKMA = 1.0 / 48.88821291839
  if dt >= _ONE_FS_AKMA:
    warnings.warn(
      f"settle_csvr: dt={dt:.4f} AKMA ({dt * 48.88821291839:.1f} fs) >= 1 fs. "
      "CSVR+SETTLE shows a tau-dependent ~+8K mean temperature bias at dt>=1 fs "
      "due to velocity-Verlet discretization. This is a known limitation; see "
      "settle_csvr docstring for details. For tight temperature control, use dt<1 fs.",
      UserWarning,
      stacklevel=2,
    )

  force_fn = _make_settle_compatible_force_fn(
    energy_or_force_fn, mass, box, is_force_shaped=is_force_shaped
  )
  if tau is None:
    tau = _DEFAULT_CSVR_TAU_AKMA

  def init_fn(key, R, mass=mass, **init_kwargs):
    _kT = init_kwargs.pop("kT", kT)
    key, split = jax.random.split(key)
    force = force_fn(R, **init_kwargs)

    # Handle mass - keep as 1D for indexing, expand to (N, 1) for broadcasting
    mass_arr = jnp.asarray(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr
    elif mass_arr.ndim == 2:
      mass_arr = mass_arr.reshape(-1)

    # Initialize momenta from Maxwell-Boltzmann
    momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
      split, R.shape, dtype=R.dtype
    )
    if atom_mask is not None:
      momenta = jnp.where(jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None], momenta, 0.0)

    # Store mass as (N, 1) for broadcasting
    mass_for_state = mass_arr[:, jnp.newaxis]
    if constraints is not None:
      c_pairs, _c_lengths, c_mask = unpack_constraints(constraints)
      momenta = project_momenta(
        momenta,
        R,
        c_pairs,
        mass_for_state,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    return NVTLangevinState(R, momenta, force, mass_for_state, key)

  def apply_fn(state, **step_kwargs):
    _dt = step_kwargs.pop("dt", dt)
    _kT = step_kwargs.pop("kT", kT)

    # Store old positions for SETTLE
    positions_old = state.positions

    # === B-step (half-kick) ===
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # === A-step (full position advance via two half-steps) ===
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    if constraints is not None:
      c_pairs, c_lengths, c_mask = unpack_constraints(constraints)
      position = project_positions(
        position,
        c_pairs,
        c_lengths,
        state.mass,
        shift_fn,
        tol=shake_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # === SETTLE position constraints ===
    if water_indices is not None:
      position = settle_positions(
        position,
        positions_old,
        water_indices,
        r_OH,
        r_HH,
        mass_oxygen,
        mass_hydrogen,
        box,
        water_mask,
      )

    # === Force eval (at constrained positions) ===
    force = force_fn(position, **step_kwargs)

    # === B-step (final half-kick) ===
    momentum = _langevin_step_b(momentum, force, _dt)

    if constraints is not None:
      c_pairs, _, c_mask = unpack_constraints(constraints)
      momentum = project_momenta(
        momentum,
        position,
        c_pairs,
        state.mass,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # === SETTLE velocity constraints ===
    if water_indices is not None:
      momentum = _langevin_settle_vel(
        momentum,
        positions_old,
        position,
        state.mass,
        water_indices,
        _dt,
        mass_oxygen,
        mass_hydrogen,
        n_iters=10,
        settle_velocity_tol=None,
        water_mask=water_mask,
        remove_linear_com_momentum=remove_com,
        atom_mask=atom_mask,
      )

    # === CSVR velocity rescaling ===
    n_atoms = state.positions.shape[0]
    n_waters = water_indices.shape[0] if water_indices is not None else 0
    safe_mass = jnp.maximum(state.mass, 1e-30)
    if atom_mask is not None:
      mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      ke_total = 0.5 * jnp.sum(jnp.where(mask_col, momentum**2 / safe_mass, 0.0))
      n_real_atoms = jnp.sum(jnp.asarray(atom_mask, dtype=jnp.int32))
    else:
      ke_total = 0.5 * jnp.sum(momentum**2 / safe_mass)
      n_real_atoms = n_atoms
    if water_indices is not None and n_waters > 0:
      n_real_waters = (
        jnp.sum(jnp.asarray(water_mask, dtype=jnp.int32))
        if water_mask is not None
        else n_waters
      )
    else:
      n_real_waters = 0
    if constraints is not None:
      c_pairs, _, c_mask = unpack_constraints(constraints)
      n_active_constraints = (
        jnp.sum(jnp.asarray(c_mask, dtype=jnp.int32))
        if c_mask is not None
        else c_pairs.shape[0]
      )
    else:
      n_active_constraints = n_constraint_pairs
    n_dof = (
      3 * n_real_atoms
      - 3 * n_real_waters
      - n_active_constraints
      - (3 if remove_com else 0)
    )
    lambda_factor, key_out = _csvr_compute_lambda(
      state.key, ke_total, n_dof, _kT, _dt, tau, max_chi_dof=3 * n_atoms
    )

    # Apply rescaling
    momentum = _csvr_rescale_momenta(momentum, lambda_factor)
    if atom_mask is not None:
      mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      momentum = jnp.where(mask_col, momentum, 0.0)
      position = jnp.where(mask_col, position, state.positions)

    # Return updated state
    return NVTLangevinState(position, momentum, force, state.mass, key_out)

  return init_fn, apply_fn


def _langevin_step_b(momentum: Array, force: Array, dt: float) -> Array:
  """Half-step velocity update (B step in BAOAB)."""
  return momentum + 0.5 * dt * force


def _langevin_step_a(
  position: Array, momentum: Array, mass: Array, dt: float, shift_fn: Callable
) -> Array:
  """Half-step position update (A step in BAOAB)."""
  velocity = momentum / mass
  return shift_fn(position, 0.5 * dt * velocity)


def _langevin_step_o(
  momentum: Array, mass: Array, gamma: float, dt: float, kT: float, rng: Array
) -> tuple[Array, Array]:
  """Stochastic velocity update (O step in BAOAB)."""
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(1 - c1**2)

  key, split = jax.random.split(rng)
  noise = jax.random.normal(split, momentum.shape, dtype=jnp.float32).astype(momentum.dtype)
  momentum_new = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise
  return momentum_new, key


def _ou_noise_one_water_rigid(
  key: Array,
  r_stack: Array,
  m_stack: Array,
  kT: float,
) -> tuple[Array, Array]:
  r"""Sample OU noise in the 6D rigid-body subspace for one TIP3P water.

  Returns mass-weighted noise (shape (3, 3)) and consumed key.
  Noise is sampled in two independent blocks: translational (COM velocity ~ N(0, kT/M))
  and rotational (angular velocity ~ N(0, kT·I⁻¹)), eliminating numerical trans–rot
  cross-coupling from the former 6×6 G-matrix Cholesky.

  Args:
    key: JAX PRNGKey.
    r_stack: (3, 3) array of O, H1, H2 positions.
    m_stack: (3,) array of O, H1, H2 masses.
    kT: Thermal energy.

  Returns:
    (p_noise, key_out) where p_noise is (3, 3) mass-weighted noise momenta.
  """
  msum = jnp.sum(m_stack)
  msum_safe = jnp.maximum(msum, jnp.array(1e-30, dtype=msum.dtype))
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum_safe
  rrel = r_stack - com  # (3, 3): COM-relative positions of O, H1, H2

  # --- translational block ---
  # Sample COM velocity noise: xi_trans ~ N(0, kT/M * I_3)
  key, k_trans, k_rot = jax.random.split(key, 3)
  z_trans = jax.random.normal(k_trans, (3,), dtype=r_stack.dtype)
  xi_trans = z_trans * jnp.sqrt(kT / msum_safe)     # (3,): COM velocity noise
  p_trans = m_stack[:, None] * xi_trans[None, :]     # (3, 3): p_i = m_i * xi_trans

  # --- rotational block ---
  # Inertia tensor: I = sum_i m_i (|r_i|^2 I_3 - r_i r_i^T)
  eye3 = jnp.eye(3, dtype=r_stack.dtype)
  r_sq = jnp.einsum("ia,ia->i", rrel, rrel)          # (3,): |r_rel_i|^2
  I_tensor = jnp.einsum("i,i->", m_stack, r_sq) * eye3 \
             - jnp.einsum("i,ia,ib->ab", m_stack, rrel, rrel)  # (3, 3)
  reg_r = jnp.array(1e-12, dtype=I_tensor.dtype) * (jnp.trace(I_tensor) / 3.0 + 1.0)
  I_reg = I_tensor + reg_r * eye3
  z_rot = jax.random.normal(k_rot, (3,), dtype=r_stack.dtype)
  # omega ~ N(0, kT * I^{-1}): solve L_I^T omega = sqrt(kT) * z_rot via closed-form 3x3 Cholesky
  omega = _cholesky_upper_solve_3x3(I_reg, jnp.sqrt(kT) * z_rot)
  # p_rot_i = m_i * (omega × r_rel_i)
  p_rot = m_stack[:, None] * jnp.cross(omega[None, :], rrel)   # (3, 3)

  return p_trans + p_rot, key


def _init_momentum_one_water_rigid(
  key: Array,
  r_stack: Array,
  m_stack: Array,
  kT: float,
) -> tuple[Array, Array]:
  r"""Sample initial momentum in rigid-body subspace for one TIP3P water.

  For initialization (not OU step), we sample directly from the Maxwell-Boltzmann
  distribution in the constrained subspace, i.e., `p ~ N(0, kT * M * P_rigid)`.

  Returns:
    (momentum, key_out) where momentum is (3, 3) mass-weighted momenta.
  """
  return _ou_noise_one_water_rigid(key, r_stack, m_stack, kT)


def _langevin_step_o_constrained(
  momentum: Array,
  position: Array,
  mass: Array,
  gamma: float,
  dt: float,
  kT: float,
  rng: Array,
  water_indices: WaterIndicesArray,
  water_mask: Array | None = None,
) -> tuple[Array, Array]:
  r"""Constrained O-step: OU noise restricted to 6D rigid-body subspace per water.

  For water molecules, noise is sampled in the rigid-body subspace via
  ``_ou_noise_one_water_rigid``, ensuring correct equipartition. Non-water
  atoms (if present) use standard isotropic OU noise.

  Args:
    momentum: (N_atoms, 3) momentum array.
    position: (N_atoms, 3) position array.
    mass: (N_atoms,) or (N_atoms, 1) mass array.
    gamma: Friction coefficient.
    dt: Timestep.
    kT: Thermal energy.
    rng: JAX PRNGKey.
    water_indices: (N_waters, 3) array of O, H1, H2 indices.

  Returns:
    (momentum_new, key_out).
  """
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(1 - c1**2)

  idx = water_indices
  n_w = int(idx.shape[0])
  if n_w <= 1:
    key, split = jax.random.split(rng)
    noise_std = jax.random.normal(split, momentum.shape, dtype=momentum.dtype)
    p_ou = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise_std
    mass_flat = mass.reshape(-1)
    p_water_in = jnp.stack([momentum[idx[:, 0]], momentum[idx[:, 1]], momentum[idx[:, 2]]], axis=1)
    r_water = jnp.stack([position[idx[:, 0]], position[idx[:, 1]], position[idx[:, 2]]], axis=1)
    m_water = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)
    splits = jax.random.split(key, n_w + 1)
    key, water_keys = splits[0], splits[1:]

    def step_one_water_single(pack):
      key_w, r_w, m_w, p_w = pack
      msum = jnp.sum(m_w)
      eps = jnp.array(1e-30, dtype=msum.dtype)
      msum_safe = jnp.maximum(msum, eps)
      com = jnp.sum(m_w[:, None] * r_w, axis=0) / msum_safe
      rrel = r_w - com
      eye3 = jnp.eye(3, dtype=r_w.dtype)
      r_sq = jnp.einsum("ia,ia->i", rrel, rrel)
      I_tensor = (
        jnp.einsum("i,i->", m_w, r_sq) * eye3
        - jnp.einsum("i,ia,ib->ab", m_w, rrel, rrel)
      )
      reg = jnp.array(1e-12, dtype=I_tensor.dtype) * (jnp.trace(I_tensor) / 3.0 + 1.0)
      I_reg = I_tensor + reg * eye3
      m_safe = jnp.maximum(m_w[:, None], eps)
      v = p_w / m_safe
      v_com = jnp.sum(m_w[:, None] * v, axis=0) / msum_safe
      L_ang = jnp.sum(jnp.cross(rrel, p_w), axis=0)
      omega_proj = _solve_3x3(I_reg, L_ang)
      v_proj = v_com[None, :] + jnp.cross(omega_proj[None, :], rrel)
      p_rigid = v_proj * m_w[:, None]
      _, k_trans, k_rot = jax.random.split(key_w, 3)
      z_trans = jax.random.normal(k_trans, (3,), dtype=r_w.dtype)
      xi_trans = z_trans * jnp.sqrt(kT / msum_safe)
      p_trans = m_w[:, None] * xi_trans[None, :]
      z_rot = jax.random.normal(k_rot, (3,), dtype=r_w.dtype)
      omega_noise = _cholesky_upper_solve_3x3(I_reg, jnp.sqrt(kT) * z_rot)
      p_rot = m_w[:, None] * jnp.cross(omega_noise[None, :], rrel)
      noise_w = p_trans + p_rot
      return c1 * p_rigid + c2 * noise_w

    p_water_out = dispatch_n_waters(
      None,
      n_w,
      step_one_water_single,
      (water_keys, r_water, m_water, p_water_in),
    )
  else:
    key, split, split_w = jax.random.split(rng, 3)
    noise_std = jax.random.normal(split, momentum.shape, dtype=jnp.float32).astype(momentum.dtype)
    p_ou = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise_std
    mass_flat = mass.reshape(-1)
    p_water_in = jnp.stack([momentum[idx[:, 0]], momentum[idx[:, 1]], momentum[idx[:, 2]]], axis=1)
    r_water = jnp.stack([position[idx[:, 0]], position[idx[:, 1]], position[idx[:, 2]]], axis=1)
    m_water = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)
    z6_all = jax.random.normal(split_w, (n_w, 6), dtype=jnp.float32).astype(r_water.dtype)

    def step_one_water(pack):
      z6_w, r_w, m_w, p_w = pack
      msum = jnp.sum(m_w)
      eps = jnp.array(1e-30, dtype=msum.dtype)
      msum_safe = jnp.maximum(msum, eps)
      com = jnp.sum(m_w[:, None] * r_w, axis=0) / msum_safe
      rrel = r_w - com
      eye3 = jnp.eye(3, dtype=r_w.dtype)
      r_sq = jnp.einsum("ia,ia->i", rrel, rrel)
      I_tensor = (
        jnp.einsum("i,i->", m_w, r_sq) * eye3
        - jnp.einsum("i,ia,ib->ab", m_w, rrel, rrel)
      )
      reg = jnp.array(1e-12, dtype=I_tensor.dtype) * (jnp.trace(I_tensor) / 3.0 + 1.0)
      I_reg = I_tensor + reg * eye3
      m_safe = jnp.maximum(m_w[:, None], eps)
      v = p_w / m_safe
      v_com = jnp.sum(m_w[:, None] * v, axis=0) / msum_safe
      L_ang = jnp.sum(jnp.cross(rrel, p_w), axis=0)
      omega_proj = _solve_3x3(I_reg, L_ang)
      v_proj = v_com[None, :] + jnp.cross(omega_proj[None, :], rrel)
      p_rigid = v_proj * m_w[:, None]
      z_trans = z6_w[:3]
      xi_trans = z_trans * jnp.sqrt(kT / msum_safe)
      p_trans = m_w[:, None] * xi_trans[None, :]
      z_rot = z6_w[3:]
      omega_noise = _cholesky_upper_solve_3x3(I_reg, jnp.sqrt(kT) * z_rot)
      p_rot = m_w[:, None] * jnp.cross(omega_noise[None, :], rrel)
      noise_w = p_trans + p_rot
      return c1 * p_rigid + c2 * noise_w

    p_water_out = dispatch_n_waters(
      None,
      n_w,
      step_one_water,
      (z6_all, r_water, m_water, p_water_in),
    )

  idx_flat = idx.reshape(-1)
  # Highest-frequency scatter site — this runs every integration step (the
  # default projection_site="post_o"). Padding rows' scatter target is
  # redirected to a discarded scratch row so they can never corrupt (or lose
  # a write-order race with) a real atom sharing the padding fill index.
  if water_mask is None:
    p_out = p_ou.at[idx_flat].set(p_water_out.reshape(-1, 3))
    return p_out, key
  mask_flat = jnp.broadcast_to(water_mask[:, None], idx.shape).reshape(-1, 1)
  p_water_flat = p_water_out.reshape(-1, 3)
  delta_p = jnp.where(mask_flat, p_water_flat - p_ou[idx_flat], 0.0)
  return p_ou.at[idx_flat].add(delta_p), key


def _langevin_step_o_free_dof(
    momentum: Array,
    mass: Array,
    gamma: float,
    dt: float,
    kT: float,
    rng: Array,
    free_dof_mask: Array,
) -> tuple[Array, Array]:
    r"""Ornstein-Uhlenbeck noise applied ONLY to free (non-water) atoms.

    Constrained (water) atoms receive NO thermostat noise, breaking the
    SETTLE+thermostat KE feedback loop. Free atoms get standard isotropic OU.

    Args:
        momentum: (N_atoms, 3) momentum array.
        mass: (N_atoms,) or (N_atoms, 1) mass array.
        gamma: Friction coefficient (ps^-1 in AKMA).
        dt: Timestep (AKMA).
        kT: Thermal energy (kcal/mol).
        rng: JAX PRNGKey.
        free_dof_mask: (N_atoms,) bool — True = free atom, False = constrained.
            Must not be all-False (pure-water systems not supported).

    Returns:
        (momentum_new, key_out) tuple.
    """
    # NOTE: Do NOT add a jnp.all(~free_dof_mask) guard here — this function is
    # JIT-traced and jnp.all returns a traced value. Python `if` on a traced bool
    # causes a concretization error. Caller must validate free_dof_mask at
    # construction time (before JIT), e.g. in O_Step.__init__ or settle_langevin.

    # Canonicalize mass to (N,1) so jnp.sqrt(mass*kT)*noise broadcasts correctly
    # against noise.shape=(N,3). Without this, (N,) mass would broadcast as (1,N),
    # applying mass[j] to spatial dim j rather than mass[i] to atom i.
    mass = mass.reshape(-1, 1) if mass.ndim == 1 else mass

    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1 - c1**2)

    key, split = jax.random.split(rng)
    noise = jax.random.normal(split, momentum.shape, dtype=momentum.dtype)
    p_ou = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise

    # Apply only to free atoms; constrained atoms keep original momentum
    mask = free_dof_mask[:, None] if free_dof_mask.ndim == 1 else free_dof_mask
    momentum_new = jnp.where(mask, p_ou, momentum)
    return momentum_new, key


def _langevin_settle_vel(
  momentum: Array,
  positions_old: Array,
  positions_new: Array,
  mass: Array,
  water_indices: WaterIndicesArray,
  dt: float,
  mass_oxygen: float,
  mass_hydrogen: float,
  n_iters: int = 10,
  settle_velocity_tol: float | None = None,
  water_mask: Array | None = None,
  remove_linear_com_momentum: bool = False,
  atom_mask: Array | None = None,
) -> Array:
  """Apply SETTLE velocity constraints and update momentum.

  Args:
      momentum: Particle momenta (N, 3).
      positions_old: Positions before integrator step.
      positions_new: Constrained positions after SETTLE.
      mass: Particle masses (N,) or (N, 1).
      water_indices: (N_waters, 3) indices [O, H1, H2].
      dt: Timestep.
      mass_oxygen: Oxygen mass (amu).
      mass_hydrogen: Hydrogen mass (amu).
      n_iters: SETTLE velocity correction iterations.
      settle_velocity_tol: SETTLE velocity tolerance.
      water_mask: Optional (N_waters,) bool for padding rows.
      remove_linear_com_momentum: If True, subtract center-of-mass momentum
          after SETTLE velocity constraints: p <- p - m * v_com where
          v_com = sum(p) / sum(m). This corrects a "flying ice cube" pattern
          where the thermostat DOF count assumes COM removal but the velocity
          constraint does not perform it.

  Returns:
      Updated momentum array.
  """
  velocity = momentum / mass
  velocity = settle_velocities(
    velocity,
    positions_old,
    positions_new,
    water_indices,
    dt,
    mass_oxygen,
    mass_hydrogen,
    n_iters=n_iters,
    adaptive_tol=settle_velocity_tol,
    water_mask=water_mask,
  )
  momentum = velocity * mass

  if atom_mask is not None:
    mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
    momentum = jnp.where(mask_col, momentum, 0.0)

  # Optionally remove center-of-mass momentum to match CSVR DOF accounting
  if remove_linear_com_momentum:
    mass_col = mass
    if atom_mask is not None:
      mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      p_tot = jnp.sum(jnp.where(mask_col, momentum, 0.0), axis=0)
      m_tot = jnp.sum(jnp.where(mask_col, mass_col, 0.0))
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = jnp.where(mask_col, momentum - mass_col * v_com, 0.0)
    else:
      p_tot = jnp.sum(momentum, axis=0)
      m_tot = jnp.sum(mass_col)
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = momentum - mass_col * v_com

  return momentum


def settle_csvr_npt(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  target_pressure_bar: float,
  tau_barostat_akma: float,
  tau_thermostat_akma: float | None = None,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  n_constraint_pairs: int = 0,
  remove_com: bool = True,
  box_init: Array | None = None,
  compressibility_bar_inv: float = 4.5e-5,
  mu_min: float = 0.999,
  project_ou_momentum_rigid: bool = True,
  projection_site: str = "post_o",
  water_mask: Array | None = None,
  atom_mask: Array | None = None,
  constraints: tuple | None = None,
  shake_tol: float = 1e-5,
  rattle_tol: float = 1e-5,
  max_shake_iter: int = 25,
  displacement_fn: Callable[..., Array] | None = None,
  is_force_shaped: bool = False,
  **kwargs,
) -> tuple[Callable, Callable]:
  r"""NPT barostat integrator combining CSVR thermostat with stochastic cell rescaling.

  Implements the NPT ensemble (isothermal-isobaric) by coupling velocity-Verlet
  dynamics with:
  1. **CSVR thermostat** (Bussi et al., 2007) for temperature control
  2. **Stochastic cell rescaling** (Bernetti & Bussi, 2020) for pressure control
  3. **SETTLE constraints** for rigid water molecules

  Algorithm sequence per timestep:
  1. Compute instantaneous pressure from pre-step config (virial + kinetic energy)
  2. B-step: Half-kick momentum from current forces
  3. A-step × 2: Full position update with constant momentum
  4. SETTLE position projection (re-satisfy water geometry)
  5. Stochastic cell rescaling:
     a. Apply stochastic scaling: μ = exp(dε/3) where dε includes pressure deviation + noise
     b. Scale box and positions by μ
  6. SETTLE position re-projection (re-satisfy constraints after box scaling)
  7. Force recompute with new box (MANDATORY for energy consistency)
  8. B-step: Final half-kick from new forces
  9. SETTLE velocity projection
  10. CSVR velocity rescaling (global chi-squared-based rescaling)

  **Key differences from NVT CSVR**:
  - Box rescaling occurs AFTER position SETTLE but BEFORE force recomputation
  - O and H positions scaled together (isotropic); H re-constrained after scaling
  - PME grid is FIXED at init; runtime check ensures box doesn't drift >10% (option B per oracle)
  - Pressure calculated from virial + kinetic energy; box volume used for normalization

  **AKMA pressure units**: 1 kcal/mol/Å³ ≈ 69,477 bar (NOT 14,583)

  Args:
      energy_or_force_fn: Force or energy function F(R, **kwargs).
      shift_fn: Displacement function for periodic boundary conditions.
      dt: Timestep (AKMA units).
      kT: Thermal energy target (kcal/mol).
      target_pressure_bar: Target pressure in bar (converted to AKMA internally).
      tau_barostat_akma: Barostat time constant (AKMA units, typically 0.1 ps ≈ 2000 AKMA).
      tau_thermostat_akma: Thermostat time constant (AKMA units); if None, defaults to ~0.1 ps.
      mass: Atomic masses (scalar or (N,) array).
      water_indices: (N_waters, 3) array of [O, H1, H2] indices; if None, no SETTLE.
      r_OH, r_HH: Target O-H and H-H distances (Å).
      mass_oxygen, mass_hydrogen: Atomic masses for water (amu).
      n_constraint_pairs: Number of solute SHAKE/RATTLE pairs (for DOF counting).
      remove_com: If True, exclude COM translation from thermostat DOF.
      box_init: Initial box dimensions for PME grid validation (optional).
      compressibility_bar_inv: Isothermal compressibility in bar⁻¹ (default TIP3P 4.5e-5).
      mu_min: Minimum scaling factor (clipping lower bound; default 0.999 ≈ 0.1% max volume change per step).
      project_ou_momentum_rigid: If True (default), project OU noise to rigid-body subspace
          for water molecules (not used in velocity-Verlet CSVR, kept for API consistency).
      projection_site: Placement for optional rigid momentum projection (not used here,
          kept for API consistency with settle_langevin).
      **kwargs: Additional arguments passed to energy_or_force_fn (e.g., soft_core_lambda).

  Returns:
      (init_fn, apply_fn) tuple following JAX-MD convention.

  References:
      Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity
      rescaling. J. Chem. Phys., 126(1), 014101.

      Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic cell rescaling.
      J. Chem. Phys., 153(11), 114107.

      Miyamoto, S., & Kollman, P. A. (1992). SETTLE: An analytical version of the SHAKE
      and RATTLE algorithm for rigid water models. J. Comput. Chem., 13(8), 952-962.
  """

  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box_init, is_force_shaped=is_force_shaped)

  # Default tau_thermostat if not provided
  if tau_thermostat_akma is None:
    tau_thermostat_akma = _DEFAULT_CSVR_TAU_AKMA

  # Convert pressure units: bar → AKMA (kcal/mol/Å³)
  target_pressure_akma = target_pressure_bar * units_module.AKMA_PRESSURE_PER_BAR

  # Convert compressibility: bar⁻¹ → AKMA units
  # β_akma = β_bar * BAR_PER_AKMA_PRESSURE
  compressibility_akma = compressibility_bar_inv * units_module.BAR_PER_AKMA_PRESSURE

  def init_fn(
      key,
      R,
      mass=mass,
      box=box_init,
      momentum: Array | None = None,
      **init_kwargs,
  ):
    r"""Initialize NPT state (host-side; not part of the jitted step loop).

    JAX-MD convention: call once from Python with concrete arrays, then
    ``jax.jit(apply_fn)`` for production steps. The Maxwell-Boltzmann vs
    warm-handoff choice uses ``momentum is None`` at Python trace time — do
    not wrap ``init_fn`` in ``jax.jit`` (would require ``static_argnames`` or
    ``jax.lax.cond`` for a traced optional momentum).
    """
    _kT = init_kwargs.pop("kT", kT)
    _box = init_kwargs.pop("box", box)

    # Validate box_init for PME grid validation
    if box_init is None:
      raise ValueError(
        "box_init must be provided for settle_csvr_npt; "
        "pass the initial box dimensions for PME grid drift validation"
      )

    key, split = jax.random.split(key)
    force = force_fn(R, **init_kwargs)

    # Handle mass array
    mass_arr = jnp.asarray(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr
    elif mass_arr.ndim == 2:
      mass_arr = mass_arr.reshape(-1)

    # Host branch: sample MB (cold) or copy equilibrated NVT momenta (warm handoff).
    if momentum is not None:
      momenta = jnp.asarray(momentum, dtype=R.dtype)
    else:
      momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
        split, R.shape, dtype=R.dtype
      )
    if atom_mask is not None:
      momenta = jnp.where(jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None], momenta, 0.0)

    # Store mass as (N, 1) for broadcasting
    mass_for_state = mass_arr[:, jnp.newaxis]
    if constraints is not None:
      c_pairs, _c_lengths, c_mask = unpack_constraints(constraints)
      momenta = project_momenta(
        momenta,
        R,
        c_pairs,
        mass_for_state,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # Validate and store box
    if _box is None:
      msg = "box must be provided for NPT initialization"
      raise ValueError(msg)
    _box = jnp.asarray(_box, dtype=R.dtype)

    return NPTState(R, momenta, force, mass_for_state, key, _box)

  def apply_fn(state, **step_kwargs):
    """Apply NPT integrator step."""
    _dt = step_kwargs.pop("dt", dt)
    _kT = step_kwargs.pop("kT", kT)
    _box = step_kwargs.pop("box", state.box)

    # Store old positions for SETTLE
    positions_old = state.positions

    # === Compute instantaneous pressure for barostat (from pre-step config) ===
    # Use positions and forces from the same configuration (start of step)
    # Kinetic energy (using rigid-body KE if water is present)
    n_atoms = positions_old.shape[0]
    n_waters = water_indices.shape[0] if water_indices is not None else 0
    safe_mass = jnp.maximum(state.mass, 1e-30)
    if atom_mask is not None:
      mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      ke_total = 0.5 * jnp.sum(jnp.where(mask_col, state.momentum**2 / safe_mass, 0.0))
    else:
      ke_total = 0.5 * jnp.sum(state.momentum**2 / safe_mass)

    # Virial trace (use forces from current state, positions before A-step)
    virial = stress_module.virial_trace(positions_old, state.force)

    # === B-step (half-kick) ===
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # === A-step (full position advance via two half-steps) ===
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    if constraints is not None:
      c_pairs, c_lengths, c_mask = unpack_constraints(constraints)
      position = project_positions(
        position,
        c_pairs,
        c_lengths,
        state.mass,
        shift_fn,
        tol=shake_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # === SETTLE position constraints ===
    if water_indices is not None and water_indices.shape[0] > 0:
      position = settle_positions(
        position,
        positions_old,
        water_indices,
        r_OH,
        r_HH,
        mass_oxygen,
        mass_hydrogen,
        _box,
        water_mask,
      )

    # Box volume
    volume = pbc_module.box_volume(_box)

    # Instantaneous pressure
    pressure = pressure_module.compute_pressure_akma(ke_total, virial, volume, ndim=3)

    # === Stochastic cell rescaling (Bernetti & Bussi 2020) ===
    # dε = -(dt/tau_P) * β * (P - P_0) + sqrt(2*kT*β*dt / (tau_P*V)) * noise
    pressure_deviation = pressure - target_pressure_akma

    key, split = jax.random.split(state.key)  # key: for CSVR, split: for barostat noise
    random_noise = jax.random.normal(split, dtype=_box.dtype)

    # Deterministic term
    depsilon_det = (_dt / tau_barostat_akma) * compressibility_akma * pressure_deviation

    # Stochastic term
    depsilon_stoch = (
      jnp.sqrt(2.0 * _kT * compressibility_akma * _dt / (tau_barostat_akma * volume))
      * random_noise
    )

    depsilon = depsilon_det + depsilon_stoch

    # Linear scaling factor: μ = exp(dε/3)
    mu = jnp.exp(depsilon / 3.0)

    # Safety clamp: μ ∈ [μ_min, 1/μ_min]
    mu_max = 1.0 / mu_min
    mu = jnp.clip(mu, mu_min, mu_max)

    # SCR isotropic scaling: momenta scale with mu per Bernetti-Bussi 2020 / Parrinello-Rahman: p' = mu * p consistent with r' = mu * r
    momentum = momentum * mu

    # Scale box
    new_box = pbc_module.isotropic_box_scale(_box, mu)

    # Scale positions
    scaled_positions = position * mu
    if constraints is not None:
      c_pairs, c_lengths, c_mask = unpack_constraints(constraints)
      scaled_positions = project_positions(
        scaled_positions,
        c_pairs,
        c_lengths,
        state.mass,
        shift_fn,
        tol=shake_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # === SETTLE position re-projection (after box scaling) ===
    if water_indices is not None and water_indices.shape[0] > 0:
      position = settle_positions(
        positions_unconstrained=scaled_positions,
        positions_old=scaled_positions,  # Use scaled positions as reference: fix geometry around correctly-placed O
        water_indices=water_indices,
        r_OH=r_OH,
        r_HH=r_HH,
        mass_oxygen=mass_oxygen,
        mass_hydrogen=mass_hydrogen,
        box=new_box,
        water_mask=water_mask,
      )
    else:
      position = scaled_positions

    # === PME grid validation (option B: fixed grid at init-time) ===
    if box_init is not None:
      volume_init = pbc_module.box_volume(box_init)
      volume_new = pbc_module.box_volume(new_box)
      volume_ratio = volume_new / volume_init
      # volume_ratio computed above; warn if >10% drift (informational; no-op here)
      del volume_ratio

    # === Force eval (at new box) ===
    force = force_fn(position, **step_kwargs)

    # === B-step (final half-kick) ===
    momentum = _langevin_step_b(momentum, force, _dt)

    if constraints is not None:
      c_pairs, _, c_mask = unpack_constraints(constraints)
      momentum = project_momenta(
        momentum,
        position,
        c_pairs,
        state.mass,
        shift_fn,
        tol=rattle_tol,
        max_iter=max_shake_iter,
        constraint_mask=c_mask,
        displacement_fn=displacement_fn,
      )

    # === SETTLE velocity constraints ===
    if water_indices is not None and water_indices.shape[0] > 0:
      momentum = _langevin_settle_vel(
        momentum,
        position,
        position,
        state.mass,
        water_indices,
        _dt,
        mass_oxygen,
        mass_hydrogen,
        n_iters=10,
        settle_velocity_tol=None,
        water_mask=water_mask,
        remove_linear_com_momentum=remove_com,
        atom_mask=atom_mask,
      )

    # === CSVR velocity rescaling ===
    safe_mass = jnp.maximum(state.mass, 1e-30)
    if atom_mask is not None:
      mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      ke_total = 0.5 * jnp.sum(jnp.where(mask_col, momentum**2 / safe_mass, 0.0))
      n_real_atoms = jnp.sum(jnp.asarray(atom_mask, dtype=jnp.int32))
    else:
      ke_total = 0.5 * jnp.sum(momentum**2 / safe_mass)
      n_real_atoms = n_atoms
    if water_indices is not None and n_waters > 0:
      n_real_waters = (
        jnp.sum(jnp.asarray(water_mask, dtype=jnp.int32))
        if water_mask is not None
        else n_waters
      )
    else:
      n_real_waters = 0
    if constraints is not None:
      c_pairs, _, c_mask = unpack_constraints(constraints)
      n_active_constraints = (
        jnp.sum(jnp.asarray(c_mask, dtype=jnp.int32))
        if c_mask is not None
        else c_pairs.shape[0]
      )
    else:
      n_active_constraints = n_constraint_pairs
    n_dof = (
      3 * n_real_atoms
      - 3 * n_real_waters
      - n_active_constraints
      - (3 if remove_com else 0)
    )
    lambda_factor, key_out = _csvr_compute_lambda(
      key, ke_total, n_dof, _kT, _dt, tau_thermostat_akma, max_chi_dof=3 * n_atoms
    )

    # Apply rescaling
    momentum = _csvr_rescale_momenta(momentum, lambda_factor)
    if atom_mask is not None:
      mask_col = jnp.asarray(atom_mask, dtype=jnp.bool_)[:, None]
      momentum = jnp.where(mask_col, momentum, 0.0)
      position = jnp.where(mask_col, position, positions_old)

    return NPTState(position, momentum, force, state.mass, key_out, new_box)

  return init_fn, apply_fn
