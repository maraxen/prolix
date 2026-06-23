"""MTT Log-Det Estimator for Coulomb-weighted Graph Laplacian.

Computes an estimate of log|det(L)| where L = D - K + εI is the regularized
Coulomb-weighted graph Laplacian, using Hutchinson trace estimation + Lanczos
tridiagonalization.

Algorithm:
  1. K_ij ≈ Φᵢ · Φⱼ via EFA features (O(ND) matvec)
  2. Lanczos: k-step tridiagonalization of L per Hutchinson probe
  3. Dense log-det of k×k tridiagonal T per probe
  4. Hutchinson average: E[z^T log(L) z] = log|det(L)|

See references/notes/mtt_theory.md for derivation and parameter guidance.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from prolix.physics.efa_coulomb import EFAParams, efa_erf_features


@dataclasses.dataclass(frozen=True)
class MTTParams:
    """Parameters for the MTT log-det estimator.

    Attributes:
        efa_params: EFAParams from efa_lebedev_params(), used to compute features.
        n_probes: Number of Hutchinson probe vectors (K in the theory doc).
        n_lanczos: Number of Lanczos steps (k in the theory doc).
        eps: Regularization shift for the graph Laplacian zero eigenvalue.
    """
    efa_params: EFAParams
    n_probes: int
    n_lanczos: int
    eps: float


def mtt_logdet_params(
    efa_params: EFAParams,
    n_probes: int = 20,
    n_lanczos: int = 50,
    eps: float = 1e-4,
) -> MTTParams:
    """Offline preprocessing: bundle EFA params + MTT hyperparameters.

    Not JAX-traced (pure Python). Call once and reuse the result.

    Args:
        efa_params: EFAParams from efa_lebedev_params().
        n_probes: Number of Hutchinson probe vectors. Variance ∝ 1/n_probes.
            Sprint 9 target: 20. Use fewer for fast testing.
        n_lanczos: Lanczos steps per probe. More steps = better approximation
            to log on the spectrum. Sprint 9 target: 50.
        eps: Regularization: L_reg = D - K + eps*I. Shifts the zero eigenvalue
            of the graph Laplacian away from 0 so log is well-defined.

    Returns:
        MTTParams dataclass.
    """
    return MTTParams(
        efa_params=efa_params,
        n_probes=n_probes,
        n_lanczos=n_lanczos,
        eps=eps,
    )


def _matvec(
    v: Float[Array, N],
    features: Float[Array, "N D"],
    eps: float,
) -> Float[Array, N]:
    """Apply the regularized graph Laplacian L_reg = D - K + εI to vector v.

    Exploits the low-rank structure K = Φ Φᵀ for O(ND) cost instead of O(N²).

    Degree diagonal: D_i = Σⱼ K_ij = Σⱼ φᵢ·φⱼ = φᵢ · (Σⱼ φⱼ).
    K @ v = features @ (features.T @ v).

    Args:
        v: (N,) vector to multiply.
        features: (N, D) EFA feature matrix Φ.
        eps: Regularization scalar.

    Returns:
        (N,) result of L_reg @ v.
    """
    # D_diag[i] = φᵢ · Σⱼ φⱼ  (correct O(ND) formula for the degree diagonal)
    phi_sum = jnp.sum(features, axis=0)      # (D,)
    D_diag = features @ phi_sum              # (N,)

    # K @ v = Φ (Φᵀ v)  — two matrix-vector products instead of one O(N²) matmul
    Kv = features @ (features.T @ v)         # (N,)

    return D_diag * v - Kv + eps * v


def _lanczos_logdet(
    matvec_fn,
    z: Float[Array, N],
    n_steps: int,
) -> Float[Array, ""]:
    """Estimate z^T log(L_reg) z using k-step Lanczos with full re-orthogonalization.

    Algorithm:
      1. Initialize q_0 = z / ||z||
      2. k steps of Lanczos recurrence with Gram-Schmidt re-orthogonalization
         against all previous vectors (full re-orth, required for accuracy).
      3. Build tridiagonal T from stored α and β entries.
      4. Compute e₁ᵀ log(T) e₁ via eigendecomposition of the small (k×k) T.
      5. Return ||z||² * e₁ᵀ log(T) e₁  (unbiased estimate of z^T log(L_reg) z).

    Args:
        matvec_fn: Closure (v,) -> L_reg @ v.
        z: (N,) Hutchinson probe vector z ~ N(0, I).
        n_steps: Number of Lanczos steps k (static — must be a Python int).

    Returns:
        Scalar estimate of z^T log(L_reg) z.
    """
    N = z.shape[0]
    z_norm = jnp.linalg.norm(z)
    q0 = z / z_norm  # (N,) — normalized starting vector

    # Pre-allocate tridiagonal coefficient arrays (fixed size for lax.scan)
    alpha_arr = jnp.zeros(n_steps)        # diagonal of T
    beta_arr = jnp.zeros(n_steps)         # sub/super-diagonal of T (beta_arr[j] is β_j)
    Q = jnp.zeros((N, n_steps))           # orthonormal basis vectors, column j = q_j

    # Store q0 as first column
    Q = Q.at[:, 0].set(q0)

    # Initial matvec
    v0 = matvec_fn(q0)
    alpha0 = jnp.dot(q0, v0)
    r0 = v0 - alpha0 * q0

    alpha_arr = alpha_arr.at[0].set(alpha0)
    beta0 = jnp.linalg.norm(r0)
    beta_arr = beta_arr.at[0].set(beta0)

    # carry: (alpha_arr, beta_arr, Q, r_prev, step_idx)
    # r_prev is the un-normalized residual from the previous step
    # step_idx tells us which column of Q we are filling next
    init_carry = (alpha_arr, beta_arr, Q, r0, jnp.int32(1))

    def lanczos_step(carry, _):
        alpha_arr, beta_arr, Q, r_prev, step = carry

        # Normalize r_prev to get next basis vector
        beta = jnp.linalg.norm(r_prev)
        # Avoid division by zero (numerical breakdown)
        safe_beta = jnp.where(beta > 1e-14, beta, 1.0)
        q_new = r_prev / safe_beta  # (N,)

        # Store into Q at column `step`
        Q = Q.at[:, step].set(q_new)

        # Matvec
        v = matvec_fn(q_new)

        # Local alpha
        alpha = jnp.dot(q_new, v)

        # Residual: remove current and previous contributions
        r = v - alpha * q_new - beta * Q[:, step - 1]

        # Full Gram-Schmidt re-orthogonalization against all stored vectors in Q.
        # Two passes for numerical stability ("twice is enough").
        coeffs = Q.T @ r        # (n_steps,) — projections onto all basis vectors
        r = r - Q @ coeffs      # (N,)       — subtract all projections (first pass)
        coeffs2 = Q.T @ r       # second pass
        r = r - Q @ coeffs2     # (N,)

        # Store tridiagonal entries
        alpha_arr = alpha_arr.at[step].set(alpha)
        # beta for THIS step goes in slot `step` (sub-diagonal entry below α[step])
        beta_arr = beta_arr.at[step].set(jnp.linalg.norm(r))

        return (alpha_arr, beta_arr, Q, r, step + 1), None

    # Run steps 1 .. n_steps-1 (step 0 was done above by hand)
    final_carry, _ = lax.scan(lanczos_step, init_carry, None, length=n_steps - 1)
    alpha_arr, beta_arr, Q, _, _ = final_carry

    # Build the k×k tridiagonal matrix T
    # T[i, i] = alpha_arr[i]
    # T[i, i+1] = T[i+1, i] = beta_arr[i]  (β_i is between steps i and i+1)
    T = jnp.diag(alpha_arr) + jnp.diag(beta_arr[:-1], k=1) + jnp.diag(beta_arr[:-1], k=-1)

    # Eigendecomposition of the small (k×k) symmetric tridiagonal T.
    # k is static (Python int), so eigh is legal here.
    eigenvals, eigenvecs = jnp.linalg.eigh(T)

    # e₁ᵀ log(T) e₁ = Σⱼ (V[0,j])² log(λⱼ)  where V = eigenvecs, λ = eigenvals
    log_eigenvals = jnp.log(jnp.maximum(eigenvals, 1e-10))
    e1_logT_e1 = jnp.sum(eigenvecs[0, :] ** 2 * log_eigenvals)

    # Unbiased estimate: z^T log(L_reg) z ≈ ||z||² * e₁ᵀ log(T) e₁
    return z_norm ** 2 * e1_logT_e1


def mtt_estimate_log_det(
    positions: Float[Array, "N 3"],
    charges: Float[Array, N],
    atom_mask: Float[Array, N],
    key: Array,
    *,
    params: MTTParams,
) -> Float[Array, ""]:
    """Stochastic estimate of log|det(L_reg)| via Hutchinson + Lanczos.

    JIT-compilable. Recommended usage:
        log_det_fn = jax.jit(functools.partial(mtt_estimate_log_det, params=params))
        log_det = log_det_fn(positions, charges, atom_mask, key)

    `params` is keyword-only so that `functools.partial(mtt_estimate_log_det,
    params=params)` returns a callable accepting (positions, charges, atom_mask, key).

    The Laplacian kernel is purely geometric (K_ij = φᵢ·φⱼ); `charges` is
    accepted for API parity and future extensions. `atom_mask` zeros ghost-atom
    feature rows so they do not contribute to K or D.

    Args:
        positions: (N, 3) atom positions in Angstrom.
        charges: (N,) partial charges. Not used in the kernel; present for API parity.
        atom_mask: (N,) float/bool mask; 0 for ghost atoms.
        key: JAX PRNG key for Hutchinson probe sampling.
        params: MTTParams from mtt_logdet_params(). Keyword-only.

    Returns:
        Scalar estimate of log|det(D - K + εI)|.
    """
    # Compute EFA feature matrix Φ ∈ R^{N × D}
    features = efa_erf_features(positions, params.efa_params)  # (N, D)

    # Zero out ghost-atom rows so they don't contribute to K or D
    atom_mask_f = jnp.asarray(atom_mask, dtype=features.dtype)
    features = features * atom_mask_f[:, None]                  # (N, D)

    eps = params.eps
    n_steps = params.n_lanczos

    # Build the matvec closure (no Python-side state captured beyond features/eps)
    def matvec(v):
        return _matvec(v, features, eps)

    # Hutchinson estimator: draw n_probes Gaussian probes, average bilinear forms.
    # Use lax.scan over probes so the loop is JIT-compiled (no Python-level loop).
    N = positions.shape[0]

    def probe_estimate(carry_key, _):
        key = carry_key
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(N,))
        estimate = _lanczos_logdet(matvec, z, n_steps)
        return key, estimate

    _, probe_estimates = lax.scan(probe_estimate, key, None, length=params.n_probes)
    # probe_estimates: (n_probes,)

    return jnp.mean(probe_estimates)
