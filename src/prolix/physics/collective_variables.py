"""Native differentiable JAX collective variables and enhanced-sampling bias potentials.

Implements:
- :class:`BiasState`: PyTree-registered frozen dataclass carrying grid bias state,
  OPES normalization ``opes_zed``, and step counter ``step`` inside ``jax.lax.scan``.
- :class:`BiasSpec`: Configuration specification for collective variables and biasing.
- Differentiable collective variables supporting ``jax.grad``, ``jax.jit``, ``jax.vmap``,
  and portable StableHLO export (zero LAPACK/cuSOLVER custom calls):
  1. :func:`cv_distance`: Minimum-image interatomic distance.
  2. :func:`cv_torsion`: 4-atom dihedral angle via pad-safe Blondel-Karplus formula.
  3. :func:`cv_radius_of_gyration`: Mass-weighted radius of gyration.
  4. :func:`cv_rmsd_kabsch`: Optimal superposition RMSD via 4x4 quaternion matrix
     repeated squaring + power iteration (100% StableHLO portable, no SVD/det).
  5. :func:`cv_plumed_kd_distance`: Exact PLUMED ``MIN`` smooth minimum distance
     (:math:`\\beta_{\\text{\\AA}} = 10.0 \\cdot \\beta_{\\text{nm}} = 200.0\\text{ \\AA}`).
- Differentiable biasing potentials and state updates:
  1. :func:`umbrella_bias_energy`: Harmonic umbrella restraint.
  2. :func:`smd_bias_energy`: Constant-velocity moving steered MD (SMD) restraint.
  3. :func:`wt_metad_bias_energy` & :func:`wt_metad_update_state`: Well-Tempered
     Metadynamics 1D cubic Hermite spline grid interpolation and Gaussian hill deposition.
  4. :func:`opes_metad_bias_energy` & :func:`opes_metad_update_state`: On-the-fly
     Probability Enhanced Sampling (OPES) KDE grid evaluation and state update.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

# Boltzmann constant in AKMA energy units: kcal / (mol * K)
BOLTZMANN_KCAL: float = 0.0019872041

DisplacementFn = Callable[[Array, Array], Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BiasState:
    """PyTree state carried inside MD integrators and ``jax.lax.scan`` loops.

    Attributes:
        bias_grid: 1D ``(G,)`` or 2D ``(G1, G2)`` bias energy grid (for ``wt_metad``)
            or kernel density estimate grid :math:`P(s)` (for ``opes_metad``).
        opes_zed: Scalar JAX array holding OPES normalization factor :math:`Z_n`.
        step: Scalar ``int32`` JAX array holding current MD step index.
        grid_min: Lower bound of the CV grid in Prolix AKMA units (Å or radians).
        grid_max: Upper bound of the CV grid in Prolix AKMA units (Å or radians).
        sigma: Gaussian hill width in CV units.
        height_kcal: Initial hill height :math:`w_0` in kcal/mol.
        bias_factor: Well-tempered / OPES bias factor :math:`\\gamma_{\\text{bias}} > 1`.
        temperature_k: Simulation temperature :math:`T` in Kelvin.
        pace: Stride (in MD steps) between hill depositions.
        opes_epsilon: Regularization floor :math:`\\epsilon_{\\text{OPES}}` in OPES log term.
        mode: Static bias mode identifier (``"umbrella"``, ``"smd"``, ``"wt_metad"``,
            ``"opes_metad"``, or ``"none"``).
    """

    bias_grid: Array
    opes_zed: Array = dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=jnp.float32)
    )
    step: Array = dataclasses.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )
    grid_min: float = 0.0
    grid_max: float = 10.0
    sigma: float = 0.2
    height_kcal: float = 0.2868
    bias_factor: float = 10.0
    temperature_k: float = 300.0
    pace: int = 500
    opes_epsilon: float = 1e-5
    mode: str = dataclasses.field(default="umbrella", metadata={"static": True})

    def replace(self, **kwargs: Any) -> BiasState:
        """Return a new :class:`BiasState` with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True)
class BiasSpec:
    """Declarative specification for collective variable biasing."""

    cv_specs: tuple[Any, ...] = ()
    bias_mode: str = "none"
    umbrella_k: float = 0.0
    umbrella_center: float = 0.0
    smd_velocity: float = 0.0
    metad_height: float = 0.2868
    metad_sigma: float = 0.2
    metad_bias_factor: float = 10.0
    metad_pace: int = 500
    plumed_min_beta_nm: float = 20.0
    grid_min: float = 0.0
    grid_max: float = 10.0
    grid_bin: int = 512
    temperature_k: float = 300.0
    opes_epsilon: float = 1e-5


def compute_opes_epsilon_from_barrier(
    barrier_kcal: float,
    bias_factor: float = 10.0,
    temperature_k: float = 300.0,
) -> float:
    """Compute OPES regularization floor epsilon = exp(-Delta E / (k_B T (1 - 1/gamma)))."""
    import math
    if barrier_kcal <= 0.0:
        return 1e-5
    gamma = max(float(bias_factor), 1.0 + 1e-6)
    kbt = BOLTZMANN_KCAL * float(temperature_k)
    denom = max(kbt * (1.0 - 1.0 / gamma), 1e-8)
    return float(max(math.exp(-float(barrier_kcal) / denom), 1e-30))


def make_bias_state_from_controls(
    controls: Any,
    *,
    grid_size: int = 512,
    grid_min: float = 0.0,
    grid_max: float = 10.0,
    temperature_k: float = 300.0,
    opes_epsilon: float | None = None,
    barrier_kcal: float | None = None,
    dtype: Any = jnp.float32,
) -> BiasState:
    """Construct a :class:`BiasState` from ``CollectiveVariableBiasControls`` or :class:`BiasSpec`."""
    temp_val = float(getattr(controls, "temperature_k", temperature_k))
    gamma_val = float(controls.metad_bias_factor)
    if opes_epsilon is not None:
        eps_val = float(opes_epsilon)
    elif hasattr(controls, "opes_epsilon") and getattr(controls, "opes_epsilon") is not None:
        eps_val = float(getattr(controls, "opes_epsilon"))
    elif barrier_kcal is not None:
        eps_val = compute_opes_epsilon_from_barrier(barrier_kcal, gamma_val, temp_val)
    else:
        eps_val = 1e-5
    return BiasState(
        bias_grid=jnp.zeros((grid_size,), dtype=dtype),
        opes_zed=jnp.array(1.0, dtype=dtype),
        step=jnp.array(0, dtype=jnp.int32),
        grid_min=float(getattr(controls, "grid_min", grid_min)),
        grid_max=float(getattr(controls, "grid_max", grid_max)),
        sigma=float(controls.metad_sigma),
        height_kcal=float(controls.metad_height),
        bias_factor=float(controls.metad_bias_factor),
        temperature_k=float(getattr(controls, "temperature_k", temperature_k)),
        pace=int(controls.metad_pace),
        opes_epsilon=eps_val,
        mode=str(controls.bias_mode),
    )


# =============================================================================
# Differentiable Collective Variables
# =============================================================================


def _apply_displacement(
    r_a: Array,
    r_b: Array,
    displacement_fn: DisplacementFn | None = None,
) -> Array:
    """Compute displacement ``r_a - r_b`` using optional PBC ``displacement_fn``."""
    if displacement_fn is None:
        return r_a - r_b
    if r_a.ndim == 1:
        return displacement_fn(r_a, r_b)
    return jax.vmap(displacement_fn)(r_a, r_b)


def cv_distance(
    positions: Array,
    atom_i: int | Array,
    atom_j: int | Array,
    displacement_fn: DisplacementFn | None = None,
) -> Array:
    """Minimum-image interatomic distance :math:`d_{ij} = \\sqrt{\\|\\Delta\\mathbf{r}_{ij}\\|^2 + 10^{-12}}`.

    Args:
        positions: Atomic coordinates of shape ``(N, 3)``.
        atom_i: First atom index (scalar int or 1D index array).
        atom_j: Second atom index (scalar int or 1D index array).
        displacement_fn: Optional PBC minimum-image displacement function.

    Returns:
        Distance in Å (scalar if ``atom_i`` is scalar, or array matching index shape).
    """
    r_i = positions[atom_i]
    r_j = positions[atom_j]
    dr = _apply_displacement(r_i, r_j, displacement_fn)
    return jnp.sqrt(jnp.sum(dr * dr, axis=-1) + 1e-12)


def cv_torsion(
    positions: Array,
    atom_indices: Array,
    displacement_fn: DisplacementFn | None = None,
) -> Array:
    """Four-atom dihedral angle :math:`\\phi \\in [-\\pi, \\pi]` via pad-safe Blondel-Karplus formula.

    Args:
        positions: Atomic coordinates of shape ``(N, 3)``.
        atom_indices: Integer array of shape ``(4,)`` or ``(K, 4)`` holding
            quadruplets ``[i, j, k, l]``.
        displacement_fn: Optional PBC minimum-image displacement function.

    Returns:
        Dihedral angle(s) in radians in :math:`[-\\pi, \\pi]`.
    """
    idx = jnp.asarray(atom_indices, dtype=jnp.int32)
    r_i = positions[idx[..., 0]]
    r_j = positions[idx[..., 1]]
    r_k = positions[idx[..., 2]]
    r_l = positions[idx[..., 3]]

    b0 = _apply_displacement(r_i, r_j, displacement_fn)
    b1 = _apply_displacement(r_k, r_j, displacement_fn)
    b2 = _apply_displacement(r_l, r_k, displacement_fn)

    b1_norm = jnp.sqrt(jnp.sum(b1 * b1, axis=-1, keepdims=True) + 1e-12)
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

    # Guard against (0, 0) backward-pass NaN on degenerate/padded quadruplets
    r2 = x * x + y * y
    is_deg = r2 < 1e-12
    x_safe = jnp.where(is_deg, 1.0, x)
    y_safe = jnp.where(is_deg, 0.0, y)
    return jnp.where(is_deg, 0.0, jnp.arctan2(y_safe, x_safe))


def cv_radius_of_gyration(
    positions: Array,
    atom_mask: Array | None = None,
    masses: Array | None = None,
) -> Array:
    """Mass-weighted radius of gyration :math:`R_g` in Å.

    .. math::

        R_g = \\sqrt{\\frac{\\sum_i m_i \\|\\mathbf{r}_i - \\mathbf{r}_{\\text{COM}}\\|^2}{\\sum_i m_i} + 10^{-12}}

    Args:
        positions: Atomic coordinates of shape ``(N, 3)``.
        atom_mask: Optional boolean mask of shape ``(N,)`` (``False`` for padded atoms).
        masses: Optional atomic masses of shape ``(N,)`` (defaults to uniform weights).

    Returns:
        Scalar radius of gyration in Å.
    """
    n_atoms = positions.shape[0]
    dtype = positions.dtype
    if masses is None:
        w = jnp.ones((n_atoms,), dtype=dtype)
    else:
        w = jnp.asarray(masses, dtype=dtype)
    if atom_mask is not None:
        w = w * jnp.asarray(atom_mask, dtype=dtype)

    w_sum = jnp.maximum(jnp.sum(w), 1e-12)
    r_com = jnp.sum(positions * w[:, None], axis=0) / w_sum
    diff = positions - r_com[None, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)
    rg_sq = jnp.sum(w * dist_sq) / w_sum
    return jnp.sqrt(rg_sq + 1e-12)


def _quaternion_to_rotation_matrix(q: Array) -> Array:
    """Convert unit quaternion ``q = [q0, q1, q2, q3]`` to a 3x3 rotation matrix."""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r00 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    r01 = 2.0 * (q1 * q2 - q0 * q3)
    r02 = 2.0 * (q1 * q3 + q0 * q2)

    r10 = 2.0 * (q2 * q1 + q0 * q3)
    r11 = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    r12 = 2.0 * (q2 * q3 - q0 * q1)

    r20 = 2.0 * (q3 * q1 - q0 * q2)
    r21 = 2.0 * (q3 * q2 + q0 * q1)
    r22 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    return jnp.array(
        [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ],
        dtype=q.dtype,
    )


def _dominant_quaternion_eigenvector(f_mat: Array) -> Array:
    """Compute dominant unit eigenvector of 4x4 symmetric Davenport matrix ``f_mat``.

    Uses pure matrix squaring + power iteration on the shifted positive-definite
    matrix ``M = F + (||F||_F + 1) I_4`` so that zero ``@lapack`` / ``cusolver``
    ``svd`` / ``eigh`` / ``det`` custom calls appear in lowered StableHLO.
    """
    dtype = f_mat.dtype
    f_norm = jnp.sqrt(jnp.sum(f_mat * f_mat) + 1e-12)
    shift = f_norm + jnp.asarray(1.0, dtype=dtype)
    m_mat = f_mat + shift * jnp.eye(4, dtype=dtype)
    m_mat = m_mat / jnp.sqrt(jnp.sum(m_mat * m_mat) + 1e-12)

    # 10 repeated squarings -> M^1024 projects onto the dominant eigenspace
    p_mat = m_mat
    for _ in range(10):
        p_mat = jnp.matmul(p_mat, p_mat)
        p_mat = p_mat / jnp.sqrt(jnp.sum(p_mat * p_mat) + 1e-12)

    # Pick the column with largest Euclidean norm (guaranteed >= 0.5 for rank-1 projector)
    col_norms_sq = jnp.sum(p_mat * p_mat, axis=0)
    best_col = jnp.argmax(col_norms_sq)
    q = p_mat[:, best_col]
    q = q / jnp.sqrt(jnp.sum(q * q) + 1e-12)

    # 8 power-iteration refinement steps
    for _ in range(8):
        q = jnp.matmul(m_mat, q)
        q = q / jnp.sqrt(jnp.sum(q * q) + 1e-12)

    return q


def cv_rmsd_kabsch(
    positions: Array,
    reference_positions: Array,
    atom_mask: Array | None = None,
) -> Array:
    """Optimal superposition RMSD via 4x4 quaternion matrix (100% StableHLO-portable).

    Avoids ``jnp.linalg.svd``, ``jnp.linalg.eigh``, and ``jnp.linalg.det`` so
    lowering inside ``prolix.export.export_context()`` emits zero LAPACK/cuSOLVER
    custom calls. By the envelope / Hellmann-Feynman theorem at the stationary
    optimal rotation :math:`\\mathbf{U}(\\mathbf{q}^*)`, ``stop_gradient(q_star)``
    yields the exact analytical position gradient :math:`\\nabla_{\\mathbf{r}} \\text{RMSD}`.

    Args:
        positions: Current atomic coordinates of shape ``(N, 3)``.
        reference_positions: Reference atomic coordinates of shape ``(N, 3)``.
        atom_mask: Optional boolean mask of shape ``(N,)`` (``False`` for padded atoms).

    Returns:
        Scalar superposition RMSD in Å.
    """
    n_atoms = positions.shape[0]
    dtype = positions.dtype
    if atom_mask is None:
        w = jnp.ones((n_atoms,), dtype=dtype)
    else:
        w = jnp.asarray(atom_mask, dtype=dtype)

    w_sum = jnp.maximum(jnp.sum(w), 1e-12)
    x_com = jnp.sum(positions * w[:, None], axis=0) / w_sum
    y_com = jnp.sum(reference_positions * w[:, None], axis=0) / w_sum

    x_c = (positions - x_com[None, :]) * w[:, None]
    y_c = (reference_positions - y_com[None, :]) * w[:, None]

    # 3x3 cross-covariance matrix R = X_c^T @ Y_c / w_sum
    r_mat = jnp.matmul(x_c.T, y_c) / w_sum
    rxx, rxy, rxz = r_mat[0, 0], r_mat[0, 1], r_mat[0, 2]
    ryx, ryy, ryz = r_mat[1, 0], r_mat[1, 1], r_mat[1, 2]
    rzx, rzy, rzz = r_mat[2, 0], r_mat[2, 1], r_mat[2, 2]

    # 4x4 symmetric Davenport / Coutsias-Seok-Dill quaternion matrix F
    f_mat = jnp.array(
        [
            [rxx + ryy + rzz, ryz - rzy, rzx - rxz, rxy - ryx],
            [ryz - rzy, rxx - ryy - rzz, rxy + ryx, rxz + rzx],
            [rzx - rxz, rxy + ryx, -rxx + ryy - rzz, ryz + rzy],
            [rxy - ryx, rxz + rzx, ryz + rzy, -rxx - ryy + rzz],
        ],
        dtype=dtype,
    )

    q_star = jax.lax.stop_gradient(_dominant_quaternion_eigenvector(f_mat))
    rot_mat = _quaternion_to_rotation_matrix(q_star)

    # Apply optimal rotation U(q^*) to centered positions
    x_aligned = jnp.matmul(positions - x_com[None, :], rot_mat.T)
    diff = x_aligned - (reference_positions - y_com[None, :])
    msd = jnp.sum(w * jnp.sum(diff * diff, axis=-1)) / w_sum
    return jnp.sqrt(msd + 1e-12)


def cv_plumed_kd_distance(
    positions: Array,
    pair_indices: Array,
    pair_mask: Array | None = None,
    displacement_fn: DisplacementFn | None = None,
    *,
    beta_nm: float = 20.0,
) -> Array:
    """PLUMED ``MIN`` smooth minimum distance :math:`d_{\\text{kd}}(\\mathbf{r})` in Å.

    Matches ``naurmalade/bias/plumed_generator.py`` (``MIN ARG=... BETA=20`` in nm,
    corresponding to :math:`\\beta_{\\text{\\AA}} = 10.0 \\cdot \\beta_{\\text{nm}} = 200.0\\text{ \\AA}`):

    .. math::

        d_{\\text{kd}}(\\mathbf{r}) = \\frac{\\beta_{\\text{\\AA}}}{\\log \\sum_k m_k \\exp(\\beta_{\\text{\\AA}} / d_{k,\\text{safe}})},
        \\qquad d_{k,\\text{safe}} = \\max(d_k, 10^{-6}\\text{ \\AA})

    Evaluated via numerically shifted log-sum-exp so :math:`\\beta_{\\text{\\AA}} / d_k` never
    overflows float32/float64 and masked-out padding pairs (:math:`m_k = 0`) contribute
    identically zero to both :math:`d_{\\text{kd}}` and :math:`\\nabla_{\\mathbf{r}} d_{\\text{kd}}`.

    Args:
        positions: Atomic coordinates of shape ``(N, 3)``.
        pair_indices: Integer array of shape ``(P, 2)`` holding atom index pairs.
        pair_mask: Optional boolean/float mask of shape ``(P,)`` (``False``/``0`` for padding).
        displacement_fn: Optional PBC minimum-image displacement function.
        beta_nm: PLUMED ``BETA`` parameter in nm (default ``20.0`` nm = ``200.0`` Å).

    Returns:
        Scalar smooth minimum distance :math:`d_{\\text{kd}}` in Å.
    """
    dtype = positions.dtype
    pairs = jnp.asarray(pair_indices, dtype=jnp.int32)
    n_pairs = pairs.shape[0]

    if pair_mask is None:
        mask_bool = jnp.ones((n_pairs,), dtype=bool)
        m_k = jnp.ones((n_pairs,), dtype=dtype)
    else:
        mask_bool = jnp.asarray(pair_mask) != 0
        m_k = jnp.where(mask_bool, jnp.asarray(pair_mask, dtype=dtype), 0.0)

    r_i = positions[pairs[:, 0]]
    r_j = positions[pairs[:, 1]]
    dr = _apply_displacement(r_i, r_j, displacement_fn)

    # Guard padded [0, 0] pairs before sqrt so padded positions have zero gradient
    dr_sq = jnp.sum(dr * dr, axis=-1)
    dr_sq_safe = jnp.where(mask_bool, dr_sq, 1.0)
    d_k = jnp.sqrt(dr_sq_safe + 1e-12)
    d_k_safe = jnp.maximum(d_k, 1e-6)

    beta_angstrom = jnp.asarray(10.0 * float(beta_nm), dtype=dtype)
    u_k = beta_angstrom / d_k_safe

    # Numerically shifted log-sum-exp over active pairs
    u_for_max = jnp.where(mask_bool, u_k, -jnp.inf)
    max_u = jax.lax.stop_gradient(jnp.max(u_for_max))
    max_u = jnp.where(jnp.isfinite(max_u), max_u, jnp.asarray(0.0, dtype=dtype))

    u_shifted = jnp.where(mask_bool, u_k - max_u, 0.0)
    exp_terms = jnp.where(mask_bool, m_k * jnp.exp(u_shifted), 0.0)
    lse = max_u + jnp.log(jnp.maximum(jnp.sum(exp_terms), 1e-30))
    return beta_angstrom / jnp.maximum(lse, 1e-12)


# =============================================================================
# Biasing Potentials & Grid Updates
# =============================================================================


def _interp_1d_catmull_rom(
    x: Array,
    grid: Array,
    grid_min: float | Array,
    grid_max: float | Array,
) -> Array:
    """Evaluate 1D C1-continuous Catmull-Rom cubic spline interpolation on a uniform grid."""
    n_bins = grid.shape[0]
    dtype = grid.dtype
    g_min = jnp.asarray(grid_min, dtype=dtype)
    g_max = jnp.asarray(grid_max, dtype=dtype)
    dx = (g_max - g_min) / jnp.maximum(jnp.asarray(n_bins - 1, dtype=dtype), 1.0)

    u = (jnp.asarray(x, dtype=dtype) - g_min) / jnp.maximum(dx, 1e-12)
    u_clamped = jnp.clip(u, 0.0, jnp.asarray(n_bins - 1, dtype=dtype))

    i1 = jnp.clip(jnp.floor(u_clamped).astype(jnp.int32), 0, max(n_bins - 2, 0))
    alpha = u_clamped - i1.astype(dtype)

    i0 = jnp.maximum(i1 - 1, 0)
    i2 = jnp.minimum(i1 + 1, n_bins - 1)
    i3 = jnp.minimum(i1 + 2, n_bins - 1)

    y0 = grid[i0]
    y1 = grid[i1]
    y2 = grid[i2]
    y3 = grid[i3]

    a0 = y1
    a1 = 0.5 * (y2 - y0)
    a2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a3 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3

    return a0 + alpha * (a1 + alpha * (a2 + alpha * a3))


def umbrella_bias_energy(
    cv_val: Array | float,
    center: Array | float,
    kappa: Array | float,
) -> Array:
    """Harmonic umbrella bias potential :math:`V_{\\text{umb}}(s) = \\frac{1}{2}\\kappa (s - s_0)^2`.

    Args:
        cv_val: Current collective variable value :math:`s(\\mathbf{r})`.
        center: Target restraint center :math:`s_0`.
        kappa: Harmonic force constant :math:`\\kappa` in kcal/(mol·CV²).

    Returns:
        Scalar umbrella bias energy in kcal/mol.
    """
    diff = jnp.asarray(cv_val) - jnp.asarray(center)
    return 0.5 * jnp.asarray(kappa) * jnp.sum(diff * diff)


def smd_bias_energy(
    cv_val: Array | float,
    initial_center: Array | float,
    kappa: Array | float,
    pulling_velocity: Array | float,
    time_or_step: Array | float,
) -> Array:
    """Moving steered MD bias :math:`V_{\\text{SMD}}(s, t) = \\frac{1}{2}\\kappa (s - (s_0 + v_{\\text{pull}} t))^2`.

    Args:
        cv_val: Current collective variable value :math:`s(\\mathbf{r})`.
        initial_center: Initial restraint center :math:`s_0` at :math:`t = 0`.
        kappa: Harmonic spring constant :math:`\\kappa` in kcal/(mol·CV²).
        pulling_velocity: Pulling rate :math:`v_{\\text{pull}}` in CV units per unit time/step.
        time_or_step: Current simulation time or step index :math:`t`.

    Returns:
        Scalar SMD bias energy in kcal/mol.
    """
    target = jnp.asarray(initial_center) + jnp.asarray(pulling_velocity) * jnp.asarray(
        time_or_step
    )
    diff = jnp.asarray(cv_val) - target
    return 0.5 * jnp.asarray(kappa) * jnp.sum(diff * diff)


def wt_metad_bias_energy(
    cv_val: Array | float,
    state: BiasState,
) -> Array:
    """Evaluate 1D Well-Tempered Metadynamics bias energy :math:`V_{\\text{meta}}(s)` via spline interpolation.

    Args:
        cv_val: Collective variable value :math:`s(\\mathbf{r})`.
        state: Current :class:`BiasState` containing ``bias_grid``.

    Returns:
        Scalar metadynamics bias energy in kcal/mol.
    """
    return _interp_1d_catmull_rom(
        jnp.asarray(cv_val),
        state.bias_grid,
        state.grid_min,
        state.grid_max,
    )


def wt_metad_update_state(
    cv_val: Array | float,
    state: BiasState,
) -> BiasState:
    """Update :class:`BiasState` for Well-Tempered Metadynamics at stride ``step % pace == 0``.

    Deposits a Gaussian hill of well-tempered height:

    .. math::

        w(s_t) = w_0 \\exp\\left(-\\frac{V(s_t)}{k_B T (\\gamma_{\\text{bias}} - 1)}\\right)

    Args:
        cv_val: Current collective variable value :math:`s_t` (treated as stop_gradient).
        state: Current :class:`BiasState`.

    Returns:
        Updated :class:`BiasState` with incremented ``step`` and updated ``bias_grid``.
    """
    dtype = state.bias_grid.dtype
    s_t = jax.lax.stop_gradient(jnp.asarray(cv_val, dtype=dtype))
    v_curr = jax.lax.stop_gradient(wt_metad_bias_energy(s_t, state))

    kbt = jnp.asarray(BOLTZMANN_KCAL, dtype=dtype) * jnp.asarray(
        state.temperature_k, dtype=dtype
    )
    kb_delta_t = jnp.maximum(
        kbt * (jnp.asarray(state.bias_factor, dtype=dtype) - 1.0),
        1e-8,
    )
    hill_height = jnp.asarray(state.height_kcal, dtype=dtype) * jnp.exp(
        -v_curr / kb_delta_t
    )

    n_bins = state.bias_grid.shape[0]
    grid_coords = jnp.linspace(
        jnp.asarray(state.grid_min, dtype=dtype),
        jnp.asarray(state.grid_max, dtype=dtype),
        n_bins,
        dtype=dtype,
    )
    sigma_safe = jnp.maximum(jnp.asarray(state.sigma, dtype=dtype), 1e-8)
    hill = hill_height * jnp.exp(-0.5 * ((grid_coords - s_t) / sigma_safe) ** 2)

    should_deposit = (state.step % state.pace) == 0
    new_grid = jnp.where(should_deposit, state.bias_grid + hill, state.bias_grid)
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    return state.replace(bias_grid=new_grid, step=new_step)


def opes_metad_bias_energy(
    cv_val: Array | float,
    state: BiasState,
) -> Array:
    """Evaluate OPES Metadynamics bias potential :math:`V_{\\text{OPES}}(s)` in kcal/mol.

    .. math::

        V_{\\text{OPES}}(s) = \\left(1 - \\frac{1}{\\gamma_{\\text{bias}}}\\right) k_B T
        \\log\\left(\\frac{P(s)}{Z_n} + \\epsilon_{\\text{OPES}}\\right)

    Args:
        cv_val: Collective variable value :math:`s(\\mathbf{r})`.
        state: Current :class:`BiasState` containing KDE ``bias_grid`` :math:`P(s)` and ``opes_zed`` :math:`Z_n`.

    Returns:
        Scalar OPES bias energy in kcal/mol.
    """
    dtype = state.bias_grid.dtype
    p_s = _interp_1d_catmull_rom(
        jnp.asarray(cv_val, dtype=dtype),
        state.bias_grid,
        state.grid_min,
        state.grid_max,
    )
    # Clamp interpolated probability density with smooth softplus-like floor so P(s)/Z_n + eps > 0
    p_s_nonneg = 0.5 * (p_s + jnp.sqrt(p_s * p_s + 1e-16))

    gamma = jnp.maximum(jnp.asarray(state.bias_factor, dtype=dtype), 1.0 + 1e-6)
    kbt = jnp.asarray(BOLTZMANN_KCAL, dtype=dtype) * jnp.asarray(
        state.temperature_k, dtype=dtype
    )
    z_n = jnp.maximum(jnp.asarray(state.opes_zed, dtype=dtype), 1e-12)
    eps = jnp.maximum(jnp.asarray(state.opes_epsilon, dtype=dtype), 1e-30)

    prefactor = (1.0 - 1.0 / gamma) * kbt
    return prefactor * jnp.log(p_s_nonneg / z_n + eps)


def opes_metad_update_state(
    cv_val: Array | float,
    state: BiasState,
) -> BiasState:
    """Update :class:`BiasState` KDE grid :math:`P(s)` and normalization :math:`Z_n` at stride ``step % pace == 0``.

    Args:
        cv_val: Current collective variable value :math:`s_t`.
        state: Current :class:`BiasState`.

    Returns:
        Updated :class:`BiasState` with incremented ``step`` and updated ``bias_grid`` and ``opes_zed``.
    """
    dtype = state.bias_grid.dtype
    s_t = jax.lax.stop_gradient(jnp.asarray(cv_val, dtype=dtype))
    p_curr = jax.lax.stop_gradient(
        _interp_1d_catmull_rom(s_t, state.bias_grid, state.grid_min, state.grid_max)
    )
    p_curr_pos = jnp.maximum(p_curr, 0.0)

    gamma = jnp.maximum(jnp.asarray(state.bias_factor, dtype=dtype), 1.0 + 1e-6)
    z_n = jnp.maximum(jnp.asarray(state.opes_zed, dtype=dtype), 1e-12)
    eps = jnp.maximum(jnp.asarray(state.opes_epsilon, dtype=dtype), 1e-30)

    # OPES reweighting weight w_t = w_0 * (p(s_t) / Z_n + eps)^(1 - 1/gamma) / eps^(1 - 1/gamma)
    exponent = 1.0 - 1.0 / gamma
    weight_t = jnp.asarray(state.height_kcal, dtype=dtype) * (
        (p_curr_pos / z_n + eps) / eps
    ) ** exponent

    n_bins = state.bias_grid.shape[0]
    grid_coords = jnp.linspace(
        jnp.asarray(state.grid_min, dtype=dtype),
        jnp.asarray(state.grid_max, dtype=dtype),
        n_bins,
        dtype=dtype,
    )
    sigma_safe = jnp.maximum(jnp.asarray(state.sigma, dtype=dtype), 1e-8)
    kernel = jnp.exp(-0.5 * ((grid_coords - s_t) / sigma_safe) ** 2)
    deposit = weight_t * kernel

    should_deposit = (state.step % state.pace) == 0
    new_grid = jnp.where(should_deposit, state.bias_grid + deposit, state.bias_grid)
    new_zed = jnp.where(
        should_deposit,
        (state.opes_zed + jnp.mean(deposit)).astype(state.opes_zed.dtype),
        state.opes_zed,
    )
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    return state.replace(bias_grid=new_grid, opes_zed=new_zed, step=new_step)




def cv_angle(
    positions: Array,
    atom_indices: Array,
    displacement_fn: DisplacementFn | None = None,
) -> Array:
    """Three-atom bond angle in radians in [0, pi]."""
    idx = jnp.asarray(atom_indices, dtype=jnp.int32)
    r_i = positions[idx[..., 0]]
    r_j = positions[idx[..., 1]]
    r_k = positions[idx[..., 2]]
    v_ji = _apply_displacement(r_i, r_j, displacement_fn)
    v_jk = _apply_displacement(r_k, r_j, displacement_fn)
    d_ji = jnp.sqrt(jnp.sum(v_ji * v_ji, axis=-1) + 1e-12)
    d_jk = jnp.sqrt(jnp.sum(v_jk * v_jk, axis=-1) + 1e-12)
    cos_theta = jnp.clip(jnp.sum(v_ji * v_jk, axis=-1) / (d_ji * d_jk + 1e-12), -0.999999, 0.999999)
    return jnp.arccos(cos_theta)


cv_rmsd = cv_rmsd_kabsch
cv_min_distance = cv_plumed_kd_distance


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ABMDState:
    """Adiabatic Bias Molecular Dynamics (ABMD) ratchet state."""

    best_cv: Array
    kappa: float = 10.0
    target_cv: float = 0.0


def abmd_bias_energy_and_update(
    cv_val: Array | float,
    state: ABMDState,
) -> tuple[Array, ABMDState]:
    """Evaluate half-harmonic ABMD ratchet bias and update best_cv toward target_cv."""
    s = jnp.asarray(cv_val)
    target = jnp.asarray(state.target_cv, dtype=s.dtype)
    dist_curr = jnp.abs(s - target)
    dist_best = jnp.abs(jnp.asarray(state.best_cv, dtype=s.dtype) - target)
    new_best_dist = jnp.minimum(jax.lax.stop_gradient(dist_curr), dist_best)
    violation = jnp.maximum(dist_curr - new_best_dist, 0.0)
    energy = 0.5 * jnp.asarray(state.kappa, dtype=s.dtype) * (violation * violation)
    new_best_cv = jnp.where(dist_curr < dist_best, jax.lax.stop_gradient(s), state.best_cv)
    return energy, dataclasses.replace(state, best_cv=new_best_cv)


def evaluate_cv_bias_from_controls(
    positions: Array,
    controls: Any,
    state: BiasState | None = None,
    displacement_fn: DisplacementFn | None = None,
    step: Array | int = 0,
) -> Array:
    """Evaluate collective variable and bias energy directly from CollectiveVariableBiasControls."""
    mode = str(controls.bias_mode).lower()
    if mode == "none" or not controls.cv_specs:
        return jnp.zeros((), dtype=positions.dtype)

    spec0 = controls.cv_specs[0]
    if isinstance(spec0, dict):
        cv_type = str(spec0.get("type", "distance")).lower()
        indices = jnp.asarray(spec0.get("indices", [0, 1]), dtype=jnp.int32)
    else:
        cv_type = "distance"
        indices = jnp.asarray(spec0, dtype=jnp.int32)

    if cv_type in ("plumed_kd_distance", "min_distance"):
        s_val = cv_plumed_kd_distance(
            positions,
            indices.reshape(-1, 2),
            displacement_fn=displacement_fn,
            beta_nm=float(controls.plumed_min_beta_nm),
        )
    elif cv_type == "torsion":
        s_val = cv_torsion(positions, indices.reshape(4), displacement_fn=displacement_fn)
    elif cv_type == "angle":
        s_val = cv_angle(positions, indices.reshape(3), displacement_fn=displacement_fn)
    elif cv_type in ("rg", "radius_of_gyration"):
        s_val = cv_radius_of_gyration(positions)
    else:
        pair = indices.reshape(-1)
        s_val = cv_distance(positions, pair[0], pair[1], displacement_fn=displacement_fn)

    if mode == "umbrella":
        return umbrella_bias_energy(s_val, float(controls.umbrella_center), float(controls.umbrella_k))
    if mode == "smd":
        return smd_bias_energy(
            s_val,
            float(controls.umbrella_center),
            float(controls.umbrella_k),
            float(controls.smd_velocity),
            step,
        )
    if mode == "wt_metad" and state is not None:
        return wt_metad_bias_energy(s_val, state)
    if mode == "opes_metad" and state is not None:
        return opes_metad_bias_energy(s_val, state)
    return jnp.zeros((), dtype=positions.dtype)


__all__ = [
    "ABMDState",
    "abmd_bias_energy_and_update",
    "compute_opes_epsilon_from_barrier",
    "cv_angle",
    "cv_min_distance",
    "cv_rmsd",
    "evaluate_cv_bias_from_controls",
    "BOLTZMANN_KCAL",
    "BiasSpec",
    "BiasState",
    "cv_distance",
    "cv_plumed_kd_distance",
    "cv_radius_of_gyration",
    "cv_rmsd_kabsch",
    "cv_torsion",
    "make_bias_state_from_controls",
    "opes_metad_bias_energy",
    "opes_metad_update_state",
    "smd_bias_energy",
    "umbrella_bias_energy",
    "wt_metad_bias_energy",
    "wt_metad_update_state",
]
