"""EFA-Lebedev Coulomb approximation for erf(αr)/r using deterministic quadrature.

Implements the Euclidean Fast Attention mechanism (Frank et al. 2025, arXiv:2412.08541)
for classical electrostatics: approximates erf(αr)/r as a sum of sinc functions
via ERoPE encoding + Lebedev quadrature on S².

Advantage over rff_coulomb.py: deterministic (zero variance). Same hybrid architecture:
  1/r = erfc(αr)/r [short-range, exact] + erf(αr)/r [long-range, this module]

See references/notes/frank2024_efa_notes.md for paper context.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from proxide.physics.constants import COULOMB_CONSTANT


@dataclass(frozen=True)
class EFAParams:
    """Cached EFA parameters for erf(αr)/r kernel approximation.

    Attributes:
        omegas: (K,) float32 array of fitted frequencies.
        weights: (K,) float32 array of fitted sinc weights.
        nodes: (G, 3) float32 array of Lebedev nodes on S².
        quad_weights: (G,) float32 array of Lebedev quadrature weights.
    """
    omegas: np.ndarray
    weights: np.ndarray
    nodes: np.ndarray
    quad_weights: np.ndarray


def _lebedev_26() -> tuple[np.ndarray, np.ndarray]:
    """26-point Lebedev quadrature on S², exact for spherical harmonics up to l=5.

    Returns:
        (nodes, weights) where:
        - nodes.shape=(26, 3), normalized to unit sphere
        - weights.shape=(26,), sum to 1; use as ∫_{S²} f(u) du ≈ 4π * sum(weights * f(nodes))
    """
    a1 = 1.0 / 21.0
    a2 = 4.0 / 105.0
    a3 = 27.0 / 840.0
    s = 1.0 / np.sqrt(2.0)
    t = 1.0 / np.sqrt(3.0)

    nodes = np.array([
        # Family 1: face centers (6 nodes)
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        # Family 2: edge midpoints (12 nodes)
        [0,  s,  s], [0,  s, -s], [0, -s,  s], [0, -s, -s],
        [ s, 0,  s], [ s, 0, -s], [-s, 0,  s], [-s, 0, -s],
        [ s,  s, 0], [ s, -s, 0], [-s,  s, 0], [-s, -s, 0],
        # Family 3: cube corners (8 nodes)
        [ t,  t,  t], [ t,  t, -t], [ t, -t,  t], [ t, -t, -t],
        [-t,  t,  t], [-t,  t, -t], [-t, -t,  t], [-t, -t, -t],
    ], dtype=np.float32)

    weights = np.array([a1]*6 + [a2]*12 + [a3]*8, dtype=np.float32)

    # Verify sum to 1
    assert np.allclose(np.sum(weights), 1.0), "Lebedev weights do not sum to 1"

    return nodes, weights


def _fit_efa_params(
    alpha: float,
    n_freqs: int,
    r_min: float = 0.3,
    r_max: float = 20.0,
    n_pts: int = 1000,
    r_eval_max: float = 8.0,
    l_lebedev: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ω_k and w_k so that Σ_k w_k sinc(ω_k r) ≈ erf(αr)/r.

    Uses non-negative least squares to ensure all weights are positive.
    Sample points are log-spaced to weight short-range and long-range equally.

    Args:
        alpha: Ewald damping parameter (1/Angstrom).
        n_freqs: Number of frequencies K.
        r_min: Minimum distance for fitting (Angstrom).
        r_max: Maximum distance for fitting (Angstrom).
        n_pts: Number of sample points in [r_min, r_max].
        r_eval_max: Maximum evaluation distance for Lebedev quadrature accuracy (Angstrom, default 8.0).
        l_lebedev: Lebedev grid exactness order (5 for 26-point grid, default 5).

    Returns:
        (omegas, weights) both shape (n_freqs,) float32.
    """
    from scipy.optimize import nnls
    from scipy.special import erf as scipy_erf

    # Log-spaced sample points give uniform density in log(r), weighting
    # short-range (where the kernel varies rapidly) and long-range equally.
    r = np.geomspace(r_min, r_max, n_pts)
    target = scipy_erf(alpha * r) / r  # shape (n_pts,)

    # Frequency grid: limit ω_max by Lebedev quadrature accuracy.
    # The 26-point Lebedev grid is exact for spherical harmonics up to l=5.
    # For the sinc approximation Σ_j λ_j cos(ω u_j·r) ≈ sinc(ωr) to hold,
    # we need ω·r_eval ≤ L_exact. This ensures NNLS weights don't need to
    # suppress large quadrature errors at higher frequencies.
    omega_max = l_lebedev / r_eval_max
    omegas = np.linspace(0.0, omega_max, n_freqs).astype(np.float32)

    # Build design matrix A[r_i, k] = sinc(ω_k * r_i)
    # sinc(0) = 1 by convention (limit as x→0 of sin(x)/x)
    phase = omegas[None, :] * r[:, None]  # (n_pts, n_freqs)
    A = np.where(phase == 0, 1.0, np.sin(phase) / phase)  # (n_pts, n_freqs)

    # Weight by 1/target so NNLS minimizes relative error uniformly over the
    # range. Without this, the fit is dominated by short-range (large values)
    # and the long-range tail (small values) is poorly approximated.
    sample_w = 1.0 / (target + 1e-8)
    weights, _ = nnls(A * sample_w[:, None], target * sample_w)
    weights = weights.astype(np.float32)

    return omegas, weights


@functools.lru_cache(maxsize=8)
def efa_lebedev_params(
    alpha: float,
    n_freqs: int = 32,
    n_lebedev_pts: int = 26,
    r_eval_max: float = 8.0,
    l_lebedev: int = 5,
) -> EFAParams:
    """Compute EFA parameters for erf(αr)/r kernel approximation.

    Results are cached — computed once per (alpha, n_freqs, n_lebedev_pts, r_eval_max, l_lebedev) tuple.

    Args:
        alpha: Ewald damping parameter (1/Angstrom). Typical: 0.34.
        n_freqs: Number of fitted frequencies K (default 32).
        n_lebedev_pts: Lebedev grid size. Currently only 26 is implemented.
        r_eval_max: Maximum evaluation distance for Lebedev quadrature accuracy (Angstrom, default 8.0).
        l_lebedev: Lebedev grid exactness order (default 5 for 26-point grid).

    Returns:
        EFAParams with precomputed omegas, weights, Lebedev nodes, and quad_weights.

    Raises:
        ValueError: If n_lebedev_pts is not 26.
    """
    if n_lebedev_pts != 26:
        raise ValueError(
            f"Only 26-point Lebedev grid is implemented; got n_lebedev_pts={n_lebedev_pts}"
        )

    nodes, quad_weights = _lebedev_26()
    omegas, weights = _fit_efa_params(alpha, n_freqs, r_eval_max=r_eval_max, l_lebedev=l_lebedev)

    return EFAParams(omegas=omegas, weights=weights, nodes=nodes, quad_weights=quad_weights)


def efa_erf_features(
    positions: Float[Array, "N 3"],
    params: EFAParams,
) -> Float[Array, "N D"]:
    """Compute EFA feature vectors for erf(αr)/r kernel.

    φ(r_i)ᵀ φ(r_j) ≈ Σ_k w_k sinc(ω_k r_ij) ≈ erf(αr)/r

    Shape: (N, 2*K*G) where K=n_freqs, G=n_lebedev_pts.

    Args:
        positions: (N, 3) atom positions.
        params: EFAParams from efa_lebedev_params().

    Returns:
        Features (N, D) with D = 2*K*G.
    """
    omegas = jnp.asarray(params.omegas)        # (K,)
    weights = jnp.asarray(params.weights)      # (K,)
    nodes = jnp.asarray(params.nodes)          # (G, 3)
    lam = jnp.asarray(params.quad_weights)    # (G,)

    # Projection: (N, G, K) <- positions (N,3) @ nodes.T (3,G) then *omegas
    proj = jnp.einsum('ni,gi->ng', positions, nodes)  # (N, G)
    proj = proj[:, :, None] * omegas[None, None, :]   # (N, G, K)

    # Feature amplitude per (node, freq): sqrt(w_k * λ_j)
    # The 4π from the Lebedev convention is already absorbed into sinc(ωr) = Σ_j λ_j cos(ω u_j·r)
    amplitude = jnp.sqrt(weights[None, :] * lam[:, None])  # (G, K)

    cos_feat = jnp.cos(proj) * amplitude[None, :, :]  # (N, G, K)
    sin_feat = jnp.sin(proj) * amplitude[None, :, :]  # (N, G, K)

    # Flatten (G, K) -> D = 2*G*K, interleave cos/sin
    N = positions.shape[0]
    features = jnp.stack([cos_feat, sin_feat], axis=-1)  # (N, G, K, 2)
    return features.reshape(N, -1)  # (N, 2*G*K)


def efa_lebedev_coulomb_energy(
    positions: Float[Array, "N 3"],
    charges: Float[Array, "N"],
    atom_mask: Float[Array, "N"],
    params: EFAParams,
) -> Float[Array, ""]:
    """O(N*K*G) erf(αr)/r Coulomb energy via EFA-Lebedev quadrature.

    E = COULOMB_CONSTANT * (||Σ_i q_i φ_i||² - Σ_i q_i² ||φ_i||²) / 2

    Approximates the long-range erf(αr)/r component. Combine with
    efa_short_range_erfc_energy from rff_coulomb.py for full Coulomb 1/r.

    Args:
        positions: (N, 3) atom positions.
        charges: (N,) partial charges.
        atom_mask: (N,) boolean mask; 0 for ghost/padded atoms.
        params: EFAParams from efa_lebedev_params().

    Returns:
        Scalar energy in kcal/mol.
    """
    charges = jnp.asarray(charges)
    atom_mask = jnp.asarray(atom_mask)
    charges_masked = charges * atom_mask

    phi = efa_erf_features(positions, params)  # (N, D)

    charge_weighted = charges_masked[:, None] * phi  # (N, D)
    charge_phi_sum = jnp.sum(charge_weighted, axis=0)  # (D,)
    quad_form = jnp.sum(charge_phi_sum ** 2)

    phi_norm_sq = jnp.sum(phi ** 2, axis=1)           # (N,)
    self_term = jnp.sum(charges_masked ** 2 * phi_norm_sq)

    energy = (quad_form - self_term) / 2.0
    return COULOMB_CONSTANT * energy


def efa_lebedev_coulomb_energy_and_params(
    positions: Float[Array, "N 3"],
    charges: Float[Array, "N"],
    atom_mask: Float[Array, "N"],
    alpha: float,
    n_freqs: int = 32,
    n_lebedev_pts: int = 26,
) -> Float[Array, ""]:
    """Convenience wrapper: compute EFA-Lebedev Coulomb energy with automatic param caching.

    This is the interface used by flash_explicit.py. Parameters are cached,
    so repeated calls with the same (alpha, n_freqs, n_lebedev_pts) reuse
    precomputed values.

    Args:
        positions: (N, 3) atom positions.
        charges: (N,) partial charges.
        atom_mask: (N,) boolean mask.
        alpha: Ewald damping parameter.
        n_freqs: Number of fitted frequencies (default 32).
        n_lebedev_pts: Lebedev grid size (default 26, only option).

    Returns:
        Scalar energy in kcal/mol.
    """
    params = efa_lebedev_params(alpha, n_freqs, n_lebedev_pts)
    return efa_lebedev_coulomb_energy(positions, charges, atom_mask, params)
