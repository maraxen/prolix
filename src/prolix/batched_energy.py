"""Batched energy function for evaluating multiple padded systems via vmap."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax_md import space
from proxide.physics.constants import COULOMB_CONSTANT

from prolix.padding import PaddedSystem
from prolix.pallas_kernels import (
    pallas_nonbonded_nl,
    resolve_pallas_mode,
)

try:
    from prolix.pallas_kernels import (
        pallas_angles_harmonic,
        pallas_bonds_2body,
        pallas_cmap_torsions,
        pallas_exceptions_14,
        pallas_gb_ace_nl,
        pallas_pme_exclusion_pairs,
        pallas_torsions_4body,
    )
except ImportError:
    pallas_angles_harmonic = None  # type: ignore[assignment]
    pallas_bonds_2body = None  # type: ignore[assignment]
    pallas_cmap_torsions = None  # type: ignore[assignment]
    pallas_exceptions_14 = None  # type: ignore[assignment]
    pallas_gb_ace_nl = None  # type: ignore[assignment]
    pallas_pme_exclusion_pairs = None  # type: ignore[assignment]
    pallas_torsions_4body = None  # type: ignore[assignment]
from prolix.physics import explicit_corrections
from prolix.physics.analytical_forces import (
    coulomb_forces_dense,
    gb_ace_forces_dense,
    lj_forces_dense,
)
from prolix.physics.bonded_masked import (
    _angle_energy_masked,
    _bond_energy_masked,
    _cmap_energy_masked,
    _dihedral_energy_masked,
)
from prolix.physics.cosolvent import cosolvent_repulsive_energy
from prolix.physics.flash_explicit import flash_explicit_forces
from prolix.physics.flash_nonbonded import flash_nonbonded_forces
from prolix.physics.generalized_born import (
    compute_ace_nonpolar_energy,
    compute_gb_energy,
    compute_gb_energy_neighbor_list,
)
from prolix.physics.neighbor_list import (
    _build_dense_exclusion_scales,
    get_neighbor_exclusion_scales,
)
from prolix.physics.optimization import chunked_coulomb_energy_nl, chunked_lj_energy_nl
from prolix.physics.pme import make_spme_energy_fn, spme_background_energy
from prolix.typing import CmapTorsionIndices

if TYPE_CHECKING:
    from jax_md.util import Array

# ==============================================================================
# POSITION RESTRAINTS (for FIRE minimization)
# ==============================================================================

def _position_restraint_energy(
    r: Array, r_ref: Array,
    k_restraint: Array, atom_mask: Array,
) -> Array:
    """Harmonic position restraint on real atoms.

    E_restraint = 0.5 * k * sum_i(|r_i - r_ref_i|^2 * mask_i)

    Used during FIRE minimization to prevent bonded geometry distortion.
    k_restraint is a JAX scalar so the same traced function works across
    all minimization stages (staged decrease from 100 to 0).

    Args:
        r: Current positions (N, 3).
        r_ref: Reference positions (N, 3) — typically the starting geometry.
        k_restraint: Spring constant (kcal/mol/Å²). JAX scalar, NOT Python float.
        atom_mask: (N,) bool mask for real atoms.
    """
    dr = r - r_ref
    dist_sq = jnp.sum(dr ** 2, axis=-1)  # (N,)
    return 0.5 * k_restraint * jnp.sum(dist_sq * atom_mask)


# ==============================================================================
# MASKED NON-BONDED TERMS (N^2 path only for now)
# ==============================================================================


def solve_ewald_alpha(cutoff_angstrom: float, error_tolerance: float = 5e-4) -> float:
    """Solve erfc(alpha * r_cut) / r_cut = delta via log-space Newton-Raphson."""
    import math

    r_cut = float(cutoff_angstrom)
    if not math.isfinite(r_cut) or r_cut <= 0.0:
        raise ValueError(f"cutoff_angstrom must be > 0, got {cutoff_angstrom}")
    delta = float(error_tolerance)
    if not math.isfinite(delta) or delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"error_tolerance must be in (0, 1), got {error_tolerance}")
    target = delta * r_cut
    if target >= 1.0:
        return 1e-6
    log_target = math.log(target)
    x = math.sqrt(max(-math.log(min(2.0 * delta, 0.999999)), 1e-12))
    for _ in range(20):
        erfc_x = math.erfc(x)
        if erfc_x <= 0.0:
            x *= 0.5
            continue
        g_val = math.log(erfc_x) - log_target
        if abs(g_val) < 1e-13:
            break
        dg_dx = -2.0 * math.exp(-(x * x)) / (math.sqrt(math.pi) * erfc_x)
        step = g_val / dg_dx
        x = max(x - max(min(step, 1.0), -0.5 * x), 1e-9)
    return float(x / r_cut)


def _lj_quintic_switch_weight(dist: Array, cutoff: float, switch_width: float) -> Array:
    """OpenMM quintic switching function S(r) = 1 - 10x^3 + 15x^4 - 6x^5 on [r_cut - w, r_cut]."""
    if switch_width <= 0.0:
        return jnp.ones_like(dist)
    r_switch = max(float(cutoff) - float(switch_width), 0.0)
    w_safe = max(float(cutoff) - r_switch, 1e-8)
    x = jnp.clip((dist - r_switch) / w_safe, 0.0, 1.0)
    s_poly = 1.0 - 10.0 * (x**3) + 15.0 * (x**4) - 6.0 * (x**5)
    return jnp.where(dist <= r_switch, 1.0, jnp.where(dist >= cutoff, 0.0, s_poly))


def _exception_energy_masked(
    r: Array,
    pairs: Array | None,
    sigmas: Array | None,
    epsilons: Array | None,
    chargeprods: Array | None,
    mask: Array | None,
    displacement_fn: space.DisplacementFn,
    *,
    soft_core_lambda: Array | None = None,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    lambda_elec: Array | float | None = None,
    soft_coulomb: bool = False,
) -> Array:
    """Explicit 1-4 exception LJ + bare Coulomb energy with optional Beutler softcore."""
    if pairs is None or pairs.shape[0] == 0:
        return jnp.array(0.0, dtype=r.dtype)
    n = r.shape[0]
    i_safe = jnp.clip(pairs[:, 0].astype(jnp.int32), 0, max(n - 1, 0))
    j_safe = jnp.clip(pairs[:, 1].astype(jnp.int32), 0, max(n - 1, 0))
    mask_f = (
        jnp.asarray(mask, dtype=r.dtype)
        if mask is not None
        else jnp.ones((pairs.shape[0],), dtype=r.dtype)
    )
    active = (mask_f > 0) & (i_safe != j_safe)
    ri = r[i_safe]
    rj = r[j_safe]
    dr = jax.vmap(displacement_fn)(ri, rj)
    dr_safe = jnp.where(active[:, None], dr, jnp.ones_like(dr))
    dist_raw = jnp.sqrt(jnp.sum(dr_safe**2, axis=-1) + jnp.asarray(1e-12, dtype=r.dtype))
    dist = jnp.where(active, dist_raw, jnp.ones_like(dist_raw))

    sig_safe = jnp.where(active, jnp.maximum(sigmas, 1e-4), 1.0)
    eps_safe = jnp.where(active, epsilons, 0.0)
    qprod_safe = jnp.where(active, chargeprods, 0.0)

    lam = (
        jnp.asarray(soft_core_lambda, dtype=r.dtype)
        if soft_core_lambda is not None
        else jnp.asarray(1.0, dtype=r.dtype)
    )
    one_minus_lam = jnp.maximum(jnp.asarray(1.0, dtype=r.dtype) - lam, 0.0)
    soft_term = jnp.asarray(softcore_alpha, dtype=r.dtype) * (one_minus_lam ** int(softcore_exponent))
    r6 = jnp.where(active, (dist / sig_safe) ** 6, jnp.ones_like(dist))
    denom = soft_term + r6 + jnp.asarray(1e-12, dtype=r.dtype)
    e_lj = 4.0 * eps_safe * lam * (1.0 / (denom * denom) - 1.0 / denom)
    lam_q = (
        jnp.asarray(lambda_elec, dtype=r.dtype)
        if lambda_elec is not None
        else jnp.asarray(1.0, dtype=r.dtype)
    )
    if soft_coulomb:
        alpha_q = jnp.asarray(0.01, dtype=r.dtype)
        denom_q = jnp.sqrt(dist * dist + alpha_q * jnp.maximum(1.0 - lam_q, 0.0) ** 2 + 1e-12)
        e_coul = COULOMB_CONSTANT * lam_q * qprod_safe / denom_q
    else:
        e_coul = COULOMB_CONSTANT * lam_q * qprod_safe / dist
    pair_e = jnp.where(active, (e_lj + e_coul) * mask_f, jnp.zeros_like(dist))
    return jnp.sum(pair_e)


def _lj_energy_masked(
    r: Array, sigmas: Array, epsilons: Array, atom_mask: Array,
    displacement_fn: space.DisplacementFn,
    soft_core_lambda: Array | None = None,
    excl_scale_vdw: Array | None = None,
    cutoff: float | None = None,
    lj_switch_width: float = 0.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
) -> Array:
    """Computes Lennard-Jones energy with atom masking and exclusion scaling.
    
    Uses soft-core Beutler (1994) formulation. When soft_core_lambda=1.0,
    the formula reduces to standard LJ. This avoids recompilation when
    varying lambda across minimization stages.

    Gradient safety: Padded atoms have sigma=0, causing (dist/sigma)^6 to
    overflow float32. Even masked by mask_ij, NaN*0=NaN in IEEE arithmetic.
    We use jnp.where to substitute safe values BEFORE the LJ computation,
    ensuring the backward pass never encounters overflow on any code path.

    Args:
        excl_scale_vdw: (N, N) scale matrix for excluded/scaled pairs.
            1.0 = full interaction, 0.0 = excluded, 0.5 = 1-4 scaled.
            If None, no exclusions are applied (legacy behavior).
    """
    n = len(sigmas)
    # Combined mask: real-atom pair AND not self-interaction
    mask_ij = (atom_mask[:, None] & atom_mask[None, :]) * (1.0 - jnp.eye(n))
    
    # Apply exclusion scales into the mask (0 = excluded, 0.5 = 1-4 scaled)
    if excl_scale_vdw is not None:
        mask_ij = mask_ij * excl_scale_vdw
    
    dr = space.map_product(displacement_fn)(r, r)
    # Gradient-safe distance: sqrt(sum(dr²) + eps) avoids d/dr sqrt(0) = NaN
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))
    
    # Mixing rules with safe values for padded atoms
    # Use jnp.where to substitute 1.0 for sigma/epsilon where mask=0,
    # preventing overflow in (dist/sigma)^6 for padded-atom pairs.
    mask_bool = mask_ij > 0.0
    sigma_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
    sigma_ij_safe = jnp.where(mask_bool, jnp.maximum(sigma_ij, jnp.float32(1e-4)), jnp.float32(1.0))
    dist_safe = jnp.where(mask_bool, jnp.maximum(dist, jnp.float32(1e-2)), jnp.float32(1.0))
    epsilon_ij = jnp.sqrt(jnp.maximum(epsilons[:, None] * epsilons[None, :], jnp.float32(0.0)))
    
    # Soft-core LJ: U(r,λ) = 4ελ[1/(α(1-λ)^p+(r/σ)⁶)² - 1/(α(1-λ)^p+(r/σ)⁶)]
    lam = jnp.float32(soft_core_lambda) if soft_core_lambda is not None else jnp.float32(1.0)
    alpha = jnp.float32(softcore_alpha)
    one_minus_lam = jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(0.0))
    soft_term = alpha * (one_minus_lam ** int(softcore_exponent))
    r_over_sig = dist_safe / sigma_ij_safe
    r6 = r_over_sig ** 6
    denom = soft_term + r6 + jnp.float32(1e-12)
    e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
        jnp.float32(1.0) / (denom * denom) - jnp.float32(1.0) / denom
    )
    if cutoff is not None and cutoff > 0.0:
        sw = _lj_quintic_switch_weight(dist, float(cutoff), float(lj_switch_width))
        e_pair = jnp.where(dist < float(cutoff), e_pair * sw, jnp.float32(0.0))
    
    # Apply combined mask (real pairs, no-self, exclusion scales)
    # Use jnp.where instead of multiplication to prevent NaN * 0 = NaN
    e_pair = jnp.where(mask_bool, e_pair * mask_ij, jnp.float32(0.0))
    
    # Divide by 2 because we double count (i,j) and (j,i)
    return 0.5 * jnp.sum(e_pair)

def _coulomb_energy_masked(
    r: Array, charges: Array, atom_mask: Array,
    displacement_fn: space.DisplacementFn,
    excl_scale_elec: Array | None = None,
    pme_alpha: float = 0.0,
    electrostatic_method: str = "pme",
    cutoff: float | None = None,
    reaction_field_dielectric: float = 78.3,
    dsf_alpha: float = 0.2,
) -> Array:
    """Computes Coulomb energy with atom masking, exclusion scaling, and PME / RF / DSF damping."""
    mask_ij = atom_mask[:, None] & atom_mask[None, :]
    
    dr = space.map_product(displacement_fn)(r, r)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))
    dist_safe = jnp.where(dist < 1e-4, 1.0, dist)
    q_prod = charges[:, None] * charges[None, :]

    method = str(electrostatic_method).lower()
    if method in ("reaction_field", "rf") and cutoff is not None and cutoff > 0.0:
        rc = float(cutoff)
        eps_rf = float(reaction_field_dielectric)
        k_rf = (eps_rf - 1.0) / ((2.0 * eps_rf + 1.0) * (rc**3))
        c_rf = (3.0 * eps_rf) / ((2.0 * eps_rf + 1.0) * rc)
        e_pair = COULOMB_CONSTANT * q_prod * (1.0 / dist_safe + k_rf * (dist_safe**2) - c_rf)
        e_pair = jnp.where(dist < rc, e_pair, 0.0)
    elif method in ("dsf", "damped_shifted_force") and cutoff is not None and cutoff > 0.0:
        from jax.scipy.special import erfc
        rc = float(cutoff)
        a_dsf = float(dsf_alpha)
        erfc_rc = erfc(a_dsf * rc)
        term_shift = erfc_rc / rc
        term_force = (
            erfc_rc / (rc * rc)
            + (2.0 * a_dsf / jnp.sqrt(jnp.pi)) * jnp.exp(-(a_dsf * a_dsf) * (rc * rc)) / rc
        )
        e_pair = COULOMB_CONSTANT * q_prod * (
            erfc(a_dsf * dist_safe) / dist_safe - term_shift + term_force * (dist_safe - rc)
        )
        e_pair = jnp.where(dist < rc, e_pair, 0.0)
    else:
        e_pair = COULOMB_CONSTANT * q_prod / dist_safe
        if pme_alpha > 0.0:
            from jax.scipy.special import erfc
            e_pair = e_pair * erfc(pme_alpha * dist_safe)
        if method == "cutoff" and cutoff is not None and cutoff > 0.0:
            e_pair = jnp.where(dist < float(cutoff), e_pair, 0.0)
    
    n = len(charges)
    no_self = 1.0 - jnp.eye(n)
    active_pair = (mask_ij > 0) & (no_self > 0)
    if excl_scale_elec is not None:
        active_pair = active_pair & (excl_scale_elec > 0)
        e_pair = jnp.where(active_pair, e_pair * mask_ij * no_self * excl_scale_elec, jnp.float32(0.0))
    else:
        e_pair = jnp.where(active_pair, e_pair * mask_ij * no_self, jnp.float32(0.0))

    return 0.5 * jnp.sum(e_pair)


def _lj_energy_neighbor_list(
    r: Array,
    sigmas: Array,
    epsilons: Array,
    neighbor_idx: Array,
    soft_core_lambda: Array | None = None,
    excl_scales_vdw: Array | None = None,
) -> Array:
    """Computes Lennard-Jones energy using neighbor list indices.

    O(N*K) scaling instead of O(N^2). Each atom i has K neighbor slots.
    Padding neighbors (idx >= N) contribute zero energy.

    Uses the same soft-core Beutler (1994) formulation as _lj_energy_masked.

    Args:
        r: Positions (N, 3).
        sigmas: LJ sigma params (N,).
        epsilons: LJ epsilon params (N,).
        neighbor_idx: Neighbor indices (N, K). Padding sentinel = N.
        soft_core_lambda: Soft-core coupling parameter. 1.0 = standard LJ.
        excl_scales_vdw: (N, K) LJ scale factors from exclusion lookup.
            1.0 = full, 0.0 = excluded, 0.5 = 1-4 scaled. If None, no exclusions.
    """
    N = r.shape[0]

    # Gather neighbor data: (N, K, 3) and (N, K)
    safe_idx = jnp.minimum(neighbor_idx, N - 1)
    r_j = r[safe_idx]                    # (N, K, 3)
    sigma_j = sigmas[safe_idx]           # (N, K)
    epsilon_j = epsilons[safe_idx]       # (N, K)

    # Central atom data broadcast: (N, 1, ...)
    r_i = r[:, None, :]                  # (N, 1, 3)
    sigma_i = sigmas[:, None]            # (N, 1)
    epsilon_i = epsilons[:, None]        # (N, 1)

    # Pairwise distances
    dr = r_i - r_j                       # (N, K, 3)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12)  # (N, K)

    # Mixing rules
    sigma_ij = 0.5 * (sigma_i + sigma_j)
    epsilon_ij = jnp.sqrt(epsilon_i * epsilon_j)

    # Soft-core LJ (same formula as _lj_energy_masked)
    lam = jnp.float32(soft_core_lambda) if soft_core_lambda is not None else jnp.float32(1.0)
    alpha = jnp.float32(0.5)
    one_minus_lam = jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(0.0))
    soft_term = alpha * (one_minus_lam ** 2)
    r_over_sig = dist / jnp.maximum(sigma_ij, jnp.float32(1e-8))
    r6 = r_over_sig ** 6
    denom = soft_term + r6 + jnp.float32(1e-12)
    e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
        jnp.float32(1.0) / (denom * denom) - jnp.float32(1.0) / denom
    )

    # Mask out padding neighbors (sentinel = N)
    mask = neighbor_idx < N  # (N, K) bool
    e_pair = jnp.where(mask, e_pair, 0.0)

    # Apply exclusion scales (1-2/1-3 zeroed, 1-4 scaled)
    if excl_scales_vdw is not None:
        e_pair = e_pair * excl_scales_vdw

    # Sum. JAX-MD NLs are symmetric (both (i,j) and (j,i) appear), so 0.5.
    return 0.5 * jnp.sum(e_pair)


def _coulomb_energy_neighbor_list(
    r: Array,
    charges: Array,
    neighbor_idx: Array,
    excl_scales_elec: Array | None = None,
) -> Array:
    """Computes Coulomb energy using neighbor list indices in O(N*K)."""
    N = r.shape[0]
    safe_idx = jnp.minimum(neighbor_idx, N - 1)
    r_j = r[safe_idx]
    r_i = r[:, None, :]
    dr = r_i - r_j
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))
    dist_safe = jnp.where(dist < 1e-4, 1.0, dist)

    q_i = charges[:, None]
    q_j = charges[safe_idx]
    e_pair = COULOMB_CONSTANT * (q_i * q_j) / dist_safe

    i_idx = jnp.arange(N, dtype=neighbor_idx.dtype)[:, None]
    mask = (neighbor_idx >= 0) & (neighbor_idx < N) & (neighbor_idx != i_idx) & (dist >= 1e-4)
    e_pair = jnp.where(mask, e_pair, 0.0)
    if excl_scales_elec is not None:
        e_pair = e_pair * excl_scales_elec
    return 0.5 * jnp.sum(e_pair)


# ==============================================================================
# BATCHED EVALUATION
# ==============================================================================

def _pme_exclusion_correction_from_pairs(
    r: Array,
    displacement_fn: space.DisplacementFn,
    excl_indices: Array,
    excl_scales_elec: Array,
    charges: Array,
    pme_alpha: float,
    n_atoms: int,
) -> Array:
    """Vmap-safe PME reciprocal-space exclusion correction.

    Removes double counting for excluded/scaled bonded pairs (same formula
    as ``explicit_corrections.pme_exclusion_correction_energy``), but reads
    per-pair scale factors directly from bundle-precomputed
    ``excl_pair_indices``/``excl_pair_scales_elec`` (populated in
    ``bundle_md.physics_system_from_bundle`` -- kept distinct from
    ``excl_indices``/``excl_scales_elec``, which since debt 765 hold the
    per-atom-row ``(N, max_excl)`` layout NL/flash kernels need, not this
    function's pair-list ``(E, 2)``/``(E,)`` shape) instead of deriving separate
    1-2/1-3/1-4 groups via ``topology.find_bonded_exclusions``. That function
    does pure Python/numpy graph traversal on ``sys.bonds`` and cannot run
    under ``jax.vmap``/``jax.jit`` tracing at all -- under EnsemblePlan's
    stacked/vmapped dispatch (any B > 1, i.e. B1's real production path),
    calling it raised ``TracerArrayConversionError`` on every real
    evaluation, silently caught by ``single_padded_energy``'s (deliberately
    broad, to also catch the one-time ``jax.eval_shape`` probe) except
    clause -- meaning this correction, and therefore PME reciprocal energy's
    entire ``if not implicit_solvent`` contribution, never executed for any
    multi-bundle stacked water-class dispatch. Confirmed via research record
    260717_pme_reciprocal_silently_disabled_under_stacked_dispatch.

    Padding rows in ``excl_indices`` are pre-redirected to index ``n_atoms``
    at bundle-construction time (see ``physics_system_from_bundle``), so the
    ``< n_atoms`` bounds check below correctly zeroes their contribution.
    """
    i_idx = excl_indices[:, 0]
    j_idx = excl_indices[:, 1]
    valid = (i_idx < n_atoms) & (j_idx < n_atoms) & (i_idx != j_idx)
    i_safe = jnp.clip(i_idx, 0, max(n_atoms - 1, 0))
    j_safe = jnp.clip(j_idx, 0, max(n_atoms - 1, 0))

    r_i = r[i_safe]
    r_j = r[j_safe]
    dr = jax.vmap(displacement_fn)(r_i, r_j)
    dist = space.distance(dr) + 1e-6

    q_i = charges[i_safe]
    q_j = charges[j_safe]
    erf_term = jax.scipy.special.erf(pme_alpha * dist)
    e_pair_recip = COULOMB_CONSTANT * (q_i * q_j / dist) * erf_term

    # excl_scales_elec is the fraction of the FULL (unscreened) interaction
    # that should remain (0.0 = fully excluded 1-2/1-3 pair, ~0.833 = scaled
    # 1-4 pair) -- the reciprocal-space contribution to REMOVE is the
    # complement of that, matching pme_exclusion_correction_energy's
    # `factor = 1.0 - scale_factor` convention exactly, just per-pair
    # instead of per-group.
    factor = jnp.where(valid, 1.0 - excl_scales_elec, 0.0)
    return -jnp.sum(e_pair_recip * factor)


def _box_for_displacement_fn(
    sys_box: Array | None,
    displacement_fn: space.DisplacementFn,
) -> Array | None:
    """Return periodic box array iff displacement_fn has a periodic closure."""
    if sys_box is not None and getattr(displacement_fn, "__closure__", None) is not None:
        return jnp.asarray(sys_box)
    return None


def _pme_reciprocal_and_corrections(
    r: Array,
    sys: PaddedSystem,
    N: int,
    e_elec: Array,
    e_lj: Array,
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool,
    direct_space_includes_exclusion_correction: bool = False,
    use_dispersion_correction: bool = True,
    use_pallas: bool | None = None,
    use_pallas_pme: bool | None = None,
) -> tuple[Array, Array]:
    """PME reciprocal-space energy + explicit-solvent corrections.

    Requires ``sys.box_size`` to be concretely readable (real FFT grid
    dimensions genuinely depend on its value, not just shape/dtype) --
    raises under ``jax.eval_shape``'s abstract tracing. Callers must catch
    that and skip this contribution when only probing output shape, since
    e_elec/e_lj's shape (always scalar) is unaffected either way. See the
    call site in ``single_padded_energy`` for why this matters (B1-NONBONDED-
    PARITY, `.praxia/docs/specs/260715_b1-nonbonded-parity.md`).

    Args:
        use_dispersion_correction: Add the isotropic LJ dispersion tail
            (OpenMM ``NonbondedForce.setUseDispersionCorrection``). Default True;
            pass False to compare against a system emitted without it.
        direct_space_includes_exclusion_correction: True when ``e_elec``'s
            direct-space term was computed by ``chunked_coulomb_energy_nl``
            (debt 760's NL path), which already sums ``pair_scale -
            erf(alpha * r)`` per pair internally -- the same self-contained
            convention ``system.py``'s ``make_energy_fn`` uses for both its
            dense and NL branches (system.py:222-226, no separate exclusion
            correction there either). ``_coulomb_energy_masked`` (the dense
            fallback here) uses a different, incomplete convention instead
            (``erfc(alpha * r) * scale``, no ``-erf`` term) that requires
            ``_pme_exclusion_correction_from_pairs`` below to become
            complete. Applying that correction unconditionally double-counts
            it on the NL branch -- root cause of debt 790 (originally
            misattributed to the NL kernels themselves; they were always
            correct, see test_nl_kernel_exclusion_parity.py).
    """

    box_arr = jnp.asarray(sys.box_size)
    mean_l = jnp.mean(box_arr.astype(jnp.float64))
    grid_points = int(sys.pme_grid_points)
    grid_spacing = float(mean_l) / float(max(grid_points, 1))
    eff_pallas_pme = use_pallas_pme if use_pallas_pme is not None else use_pallas
    pme_recip_fn = make_spme_energy_fn(
        box_size=box_arr,
        alpha=float(sys.pme_alpha),
        grid_spacing=grid_spacing,
        use_pallas_pme=eff_pallas_pme,
    )
    e_elec = e_elec + pme_recip_fn(r, sys.charges, sys.atom_mask)
    e_elec = e_elec + spme_background_energy(
        sys.charges, sys.atom_mask, sys.pme_alpha, sys.box_size
    )

    if not implicit_solvent:
        safe_charges = jnp.where(sys.atom_mask, sys.charges, jnp.float32(0.0))
        # excl_pair_indices/excl_pair_scales_elec (pair-list, (E,2)/(E,)) --
        # NOT excl_indices/excl_scales_elec, which since debt 765 hold the
        # per-atom-row (N, max_excl) layout NL/flash kernels need.
        if (
            not direct_space_includes_exclusion_correction
            and sys.excl_pair_indices is not None
            and sys.excl_pair_indices.shape[0] > 0
        ):
            pallas_enabled, interpret = resolve_pallas_mode(use_pallas=use_pallas)
            if pallas_enabled and pallas_pme_exclusion_pairs is not None:
                e_elec = e_elec + pallas_pme_exclusion_pairs(
                    r,
                    sys.excl_pair_indices,
                    sys.excl_pair_scales_elec,
                    safe_charges,
                    pme_alpha=float(sys.pme_alpha),
                    n_atoms=N,
                    box=_box_for_displacement_fn(sys.box_size, displacement_fn),
                    atom_mask=None,
                    is_row_format=False,
                    interpret=interpret,
                )
            else:
                e_elec = e_elec + _pme_exclusion_correction_from_pairs(
                    r,
                    displacement_fn,
                    sys.excl_pair_indices,
                    sys.excl_pair_scales_elec,
                    safe_charges,
                    float(sys.pme_alpha),
                    N,
                )
        safe_sigmas = jnp.where(sys.atom_mask, sys.sigmas, jnp.float32(1.0))
        safe_epsilons = jnp.where(sys.atom_mask, sys.epsilons, jnp.float32(0.0))
        # lj_dispersion_tail_energy's last parameter is `atom_mask` (an (N,)
        # array) -- this used to pass a scalar real-atom COUNT (n_tail)
        # instead. Since the count gets squared inside the C6/C12 sums, the
        # error scales as n_atoms^2: negligible for B1's 12-atom water class
        # (~144x on an already-tiny term) but catastrophic for a real
        # solvated protein (~1963 atoms -> ~3.85e6x, a multi-million-kcal/mol
        # energy from what should be a small long-range correction). Found
        # via the solvate_protein_to_bundle adapter verification against the
        # already-tested pad_solvated_system/single_padded_energy_nl_cvjp
        # reference pathway (1VII solvated, 32A box).
        if use_dispersion_correction:
            e_lj = e_lj + explicit_corrections.lj_dispersion_tail_energy(
                box_arr,
                safe_sigmas,
                safe_epsilons,
                float(sys.nonbonded_cutoff),
                sys.atom_mask,
            )

    return e_elec, e_lj


def single_padded_energy(
    sys: PaddedSystem,
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    soft_core_lambda: Array = jnp.array(1.0),
    neighbor: Any = None,
    lj_switch_width: float = 0.0,
    use_dispersion_correction: bool = True,
    use_pallas: bool | None = None,
    use_pallas_pme: bool | None = None,
) -> Array:
    """Computes total potential energy for a single padded system.

    This is the public API for computing energy of one PaddedSystem.
    It can be used standalone with jax.grad, or batched via make_batched_energy_fn.

    With periodic boundaries (``box_size`` + positive ``pme_alpha``) and
    ``implicit_solvent=False``, reciprocal PME uses the same grid-spacing policy
    as ``physics.system.make_energy_fn`` and ``physics.flash_explicit`` (mean box
    edge / ``pme_grid_points``), and adds the same PME exclusion correction and
    isotropic LJ dispersion tail as those paths (see ``explicit_corrections``).

    Args:
        soft_core_lambda: JAX array for soft-core LJ coupling.
            λ=1.0 → standard LJ, λ<1.0 → soft-core (for staged minimization).
            None defaults to λ=1.0 (standard LJ).
        use_dispersion_correction: Add the isotropic LJ dispersion tail on
            the periodic PME path (OpenMM ``setUseDispersionCorrection``).
        neighbor: Optional jax_md ``NeighborList`` (or a raw ``(N, K)`` index
            array, matching ``physics.system.make_energy_fn``'s
            ``getattr(neighbor, "idx", neighbor)`` convention) — when given,
            the direct-space LJ/Coulomb terms are computed via
            ``optimization.chunked_lj_energy_nl``/``chunked_coulomb_energy_nl``
            (O(N*K)) instead of the dense O(N²) masked path (debt 760: no
            PME-over-neighbor-list Coulomb term existed anywhere in the
            bundle world before this — this mirrors ``system.py``'s own
            ``electrostatics_energy_fn_bound`` NL branch exactly, not new
            physics). Requires ``sys.excl_indices``/``excl_scales_vdw``/
            ``excl_scales_elec`` to be the per-atom-row ``(N, max_excl)``
            layout (debt 765) — always true for bundle-derived systems as of
            that fix.

            debt 790 (filed 2026-07-17, root-caused and fixed 2026-07-18):
            this branch was initially blocked by a confirmed, wildly-wrong
            energy under real PME exclusions. Root cause was NOT the NL
            kernels (``chunked_lj_energy_nl``/``chunked_coulomb_energy_nl``
            are self-contained and correct — they already sum ``pair_scale -
            erf(alpha * r)`` per pair internally, the same convention
            ``system.py``'s already-validated NL branch uses). It was
            ``_pme_reciprocal_and_corrections`` unconditionally adding the
            separate ``_pme_exclusion_correction_from_pairs`` term needed to
            complete the dense path's different, incomplete convention
            (``_coulomb_energy_masked`` uses ``erfc(alpha*r) * scale``, no
            ``-erf`` term) — double-counting that correction on the NL
            branch, whose direct-space term already includes it. Fixed by
            passing ``direct_space_includes_exclusion_correction=(neighbor
            is not None)`` into ``_pme_reciprocal_and_corrections``. See
            ``tests/physics/test_nl_kernel_exclusion_parity.py`` (permanent
            regression test) and
            ``.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md``
            for the full bisection trail. PME reciprocal + LJ-tail (below)
            are unaffected and identical either way — only the exclusion
            correction is now conditional on which direct-space path ran.

            (Original filing misdiagnosed the NL kernels themselves as
            broken, based on a comparison against ``_coulomb_energy_masked``
            in isolation — which is itself incomplete-by-design and only
            correct when paired with the exclusion correction it was never
            given in that isolated test. Ghost/padding atoms and the
            exclusion data itself were independently ruled out and remain
            ruled out; see the decision doc for the full trail.)
    """
    r = sys.positions
    pallas_enabled, interpret = resolve_pallas_mode(use_pallas=use_pallas)
    box_pbc = _box_for_displacement_fn(sys.box_size, displacement_fn)
    _has_pallas_bonded = (
        pallas_bonds_2body is not None
        and pallas_angles_harmonic is not None
        and pallas_torsions_4body is not None
        and pallas_cmap_torsions is not None
    )

    # Bonded terms
    if pallas_enabled and _has_pallas_bonded:
        with jax.named_scope("dense_bonded_bond"):
            e_bond = pallas_bonds_2body(
                r, sys.bonds, sys.bond_params, sys.bond_mask,
                box=box_pbc, dist_eps=1e-10, interpret=interpret,
            )
        with jax.named_scope("dense_bonded_urey_bradley"):
            e_ub = pallas_bonds_2body(
                r, sys.urey_bradley_bonds, sys.urey_bradley_params, sys.urey_bradley_mask,
                box=box_pbc, dist_eps=1e-10, interpret=interpret,
            )
        with jax.named_scope("dense_bonded_angle"):
            e_angle = pallas_angles_harmonic(
                r, sys.angles, sys.angle_params, sys.angle_mask,
                box=box_pbc, dist_eps=1e-10, interpret=interpret,
            )
        with jax.named_scope("dense_bonded_dihedral"):
            e_dih = pallas_torsions_4body(
                r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask,
                box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
        with jax.named_scope("dense_bonded_improper"):
            e_imp = pallas_torsions_4body(
                r, sys.impropers, sys.improper_params, sys.improper_mask,
                box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
        with jax.named_scope("dense_bonded_cmap"):
            e_cmap = pallas_cmap_torsions(
                r, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs,
                box=box_pbc, first_is_psi=True, interpret=interpret,
            )
    else:
        with jax.named_scope("dense_bonded_bond"):
            e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
        with jax.named_scope("dense_bonded_urey_bradley"):
            e_ub = _bond_energy_masked(r, sys.urey_bradley_bonds, sys.urey_bradley_params, sys.urey_bradley_mask, displacement_fn)
        with jax.named_scope("dense_bonded_angle"):
            e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
        with jax.named_scope("dense_bonded_dihedral"):
            e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
        with jax.named_scope("dense_bonded_improper"):
            e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
        with jax.named_scope("dense_bonded_cmap"):
            e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)

    N = len(sys.atom_mask)

    if neighbor is not None:
        nb_idx = getattr(neighbor, "idx", neighbor)
        nb_idx = jnp.where(
            sys.atom_mask[:, None] & sys.atom_mask[jnp.minimum(nb_idx, N - 1)],
            nb_idx,
            N,
        )
        if pallas_enabled and pallas_nonbonded_nl is not None:
            e_lj = pallas_nonbonded_nl(
                r,
                sys.charges,
                sys.sigmas,
                sys.epsilons,
                sys.atom_mask,
                nb_idx,
                sys.excl_indices,
                sys.excl_scales_vdw,
                sys.excl_scales_elec,
                box=box_pbc,
                pme_alpha=float(sys.pme_alpha),
                cutoff=float(sys.nonbonded_cutoff),
                lj_switch_width=float(lj_switch_width),
                soft_core_lambda=soft_core_lambda,
                include_pme_excl_erf=True,
                interpret=interpret,
            )
            e_elec = jnp.asarray(0.0, dtype=r.dtype)
        else:
            e_lj = chunked_lj_energy_nl(
                r, sys.sigmas, sys.epsilons, sys.excl_indices, sys.excl_scales_vdw,
                nb_idx, displacement_fn, float(sys.nonbonded_cutoff), 128,
                float(lj_switch_width),
            )
            e_elec = chunked_coulomb_energy_nl(
                r, sys.charges, sys.excl_indices, sys.excl_scales_elec,
                nb_idx, displacement_fn, float(sys.pme_alpha), 332.0637,
                float(sys.nonbonded_cutoff), 128,
            )
    else:
        # Dense O(N^2) path — use precomputed dense exclusion scale matrices
        # if available. These are topology constants (independent of r).
        # Precomputing once avoids the costly fori_loop+scatter rebuild every
        # step.
        # stop_gradient is critical: the fori_loop+scatter backward pass in
        # _build_dense_exclusion_scales produces NaN due to overlapping
        # scatter indices. These matrices are topology constants
        # (independent of r), so cutting the gradient is both correct and
        # necessary.
        if sys.dense_excl_scale_vdw is not None:
            excl_scale_vdw = sys.dense_excl_scale_vdw
        else:
            excl_scale_vdw = jax.lax.stop_gradient(_build_dense_exclusion_scales(
                sys.excl_indices, sys.excl_scales_vdw, N,
            ))
        if sys.dense_excl_scale_elec is not None:
            excl_scale_elec = sys.dense_excl_scale_elec
        else:
            excl_scale_elec = jax.lax.stop_gradient(_build_dense_exclusion_scales(
                sys.excl_indices, sys.excl_scales_elec, N,
            ))

        elec_method = str(getattr(sys, "electrostatic_method", "pme")).lower()
        with jax.named_scope("dense_lj_direct"):
            eff_switch = float(lj_switch_width) if lj_switch_width > 0.0 else float(getattr(sys, "lj_switch_width", 0.0))
            raw_cutoff = float(getattr(sys, "nonbonded_cutoff", 0.0))
            needs_cutoff = (
                sys.box_size is not None
                or eff_switch > 0.0
                or elec_method in ("reaction_field", "rf", "dsf", "damped_shifted_force", "cutoff")
            )
            eff_cutoff = raw_cutoff if (needs_cutoff and raw_cutoff > 0.0) else None
            e_lj = _lj_energy_masked(
                r, sys.sigmas, sys.epsilons, sys.atom_mask, displacement_fn,
                soft_core_lambda=soft_core_lambda, excl_scale_vdw=excl_scale_vdw,
                cutoff=eff_cutoff, lj_switch_width=eff_switch,
                softcore_alpha=float(getattr(sys, "softcore_alpha", 0.5)),
                softcore_exponent=int(getattr(sys, "softcore_exponent", 2)),
            )

        # Coulomb interaction (direct space, optionally damped for PME / RF / DSF)
        with jax.named_scope("dense_coulomb_direct"):
            elec_method = str(getattr(sys, "electrostatic_method", "pme")).lower()
            e_elec = _coulomb_energy_masked(
                r, sys.charges, sys.atom_mask, displacement_fn,
                excl_scale_elec=excl_scale_elec,
                pme_alpha=sys.pme_alpha if elec_method == "pme" else 0.0,
                electrostatic_method=elec_method,
                cutoff=eff_cutoff,
                reaction_field_dielectric=float(getattr(sys, "reaction_field_dielectric", 78.3)),
                dsf_alpha=float(getattr(sys, "dsf_alpha", 0.2)),
            )

    # Explicit 1-4 exception energy (FR-6 / CA-3)
    if pallas_enabled and pallas_exceptions_14 is not None:
        e_exc = pallas_exceptions_14(
            r,
            getattr(sys, "exception_pairs", None),
            getattr(sys, "exception_sigmas", None),
            getattr(sys, "exception_epsilons", None),
            getattr(sys, "exception_chargeprods", None),
            getattr(sys, "exception_mask", None),
            box=box_pbc,
            soft_core_lambda=soft_core_lambda,
            softcore_alpha=float(getattr(sys, "softcore_alpha", 0.5)),
            softcore_exponent=int(getattr(sys, "softcore_exponent", 2)),
            interpret=interpret,
        )
    else:
        e_exc = _exception_energy_masked(
            r,
            getattr(sys, "exception_pairs", None),
            getattr(sys, "exception_sigmas", None),
            getattr(sys, "exception_epsilons", None),
            getattr(sys, "exception_chargeprods", None),
            getattr(sys, "exception_mask", None),
            displacement_fn,
            soft_core_lambda=soft_core_lambda,
            softcore_alpha=float(getattr(sys, "softcore_alpha", 0.5)),
            softcore_exponent=int(getattr(sys, "softcore_exponent", 2)),
        )
    e_lj = e_lj + e_exc

    # Reciprocal Space (PME) + explicit-solvent corrections (aligned with system / Flash)
    elec_method = str(getattr(sys, "electrostatic_method", "pme")).lower()
    eff_disp_corr = bool(use_dispersion_correction and getattr(sys, "use_dispersion_correction", True))
    if sys.box_size is not None and sys.pme_alpha > 0.0 and elec_method == "pme":
        try:
            with jax.named_scope("dense_pme_reciprocal"):
                e_elec, e_lj = _pme_reciprocal_and_corrections(
                    r, sys, N, e_elec, e_lj, displacement_fn, implicit_solvent,
                    direct_space_includes_exclusion_correction=(neighbor is not None),
                    use_dispersion_correction=eff_disp_corr,
                    use_pallas=use_pallas,
                    use_pallas_pme=use_pallas_pme,
                )
        except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
            pass
    elif sys.box_size is not None and not implicit_solvent and eff_disp_corr:
        try:
            box_arr = jnp.asarray(sys.box_size)
            safe_sigmas = jnp.where(sys.atom_mask, sys.sigmas, jnp.float32(1.0))
            safe_epsilons = jnp.where(sys.atom_mask, sys.epsilons, jnp.float32(0.0))
            e_lj = e_lj + explicit_corrections.lj_dispersion_tail_energy(
                box_arr, safe_sigmas, safe_epsilons, float(sys.nonbonded_cutoff), sys.atom_mask
            )
        except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
            pass

    if implicit_solvent:
        mask_ij = sys.atom_mask[:, None] & sys.atom_mask[None, :]
        energy_mask = mask_ij * (1.0 - jnp.eye(N))
        e_gb, born_radii = compute_gb_energy(
            positions=r,
            charges=sys.charges,
            radii=sys.radii,
            scaled_radii=sys.scaled_radii,
            mask=sys.atom_mask,
            energy_mask=energy_mask,
            solvent_dielectric=float(getattr(sys, "solvent_dielectric", 78.5)),
            solute_dielectric=float(getattr(sys, "solute_dielectric", 1.0)),
            dielectric_offset=float(getattr(sys, "dielectric_offset", 0.09)),
        )
        e_np = compute_ace_nonpolar_energy(
            sys.radii,
            born_radii,
            surface_tension=float(getattr(sys, "surface_tension", 0.0054)),
            dielectric_offset=float(getattr(sys, "dielectric_offset", 0.09)),
        )
        # Mask out nonpolar energy for padding atoms (ACE is per-atom)
        e_np = jnp.sum(e_np * sys.atom_mask)
        e_solv = e_gb + e_np
    else:
        e_solv = 0.0

    e_cosolv_rep = jnp.float32(0.0)
    if getattr(sys, "cosolvent_group_id", None) is not None and getattr(sys, "cosolvent_mol_id", None) is not None:
        e_cosolv_rep = cosolvent_repulsive_energy(
            r,
            sys.cosolvent_group_id,
            sys.cosolvent_mol_id,
            displacement_fn,
            eps_kcal=float(getattr(sys, "cosolvent_repulsive_eps", 0.01)),
            r0_angstrom=float(getattr(sys, "cosolvent_repulsive_r0", 10.0)),
            cutoff_angstrom=float(getattr(sys, "cosolvent_repulsive_cutoff", 9.0)),
            deduplicate_interaction_groups=bool(getattr(sys, "deduplicate_interaction_groups", False)),
            atom_mask=sys.atom_mask,
        )

    return e_bond + e_ub + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv + e_cosolv_rep


def single_padded_force(
    sys: PaddedSystem,
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    explicit_solvent: bool = False,
    soft_core_lambda: Array = jnp.array(1.0),
    use_flash: bool = True,
    flash_tile_size: int = 256,
    flash_remat: bool = True,
    use_pallas: bool | None = None,
) -> Array:
    """Compute forces for a padded system.

    Two paths:
      - FlashMD (use_flash=True, default): Chunked+checkpointed nonbonded
        energy via tiled O(N²) with sparse exclusions. No dense
        (N,N) exclusion matrices needed — memory is O(N) per replica.
      - Legacy (use_flash=False): Separate analytical LJ, Coulomb, GB
        force functions using dense (N,N) exclusion matrices.

    Bonded terms (bonds, angles, dihedrals, impropers, CMAP) are always
    computed via jax.grad — safe because they use indexed gathers (no N²).

    Args:
        sys: A PaddedSystem with all force field parameters.
        displacement_fn: JAX-MD displacement function.
        implicit_solvent: Whether to include GB solvation.
        soft_core_lambda: Soft-core coupling for LJ. 1.0 = standard.
        use_flash: Use FlashMD architecture (default True). Set False to
            use legacy dense-matrix path (requires dense_excl_scale_* fields).

    Returns:
        Forces (N, 3) in kcal/mol/Å. F = -dE/dr.
    """
    r = sys.positions
    N = len(sys.atom_mask)
    pallas_enabled, interpret = resolve_pallas_mode(use_pallas=use_pallas)
    box_pbc = _box_for_displacement_fn(sys.box_size, displacement_fn)
    _has_pallas_bonded_force = (
        pallas_bonds_2body is not None
        and pallas_angles_harmonic is not None
        and pallas_torsions_4body is not None
        and pallas_cmap_torsions is not None
        and pallas_exceptions_14 is not None
    )

    # --- Bonded energy only (safe for jax.grad — no N² operations) ---
    def bonded_energy(positions):
        if pallas_enabled and _has_pallas_bonded_force:
            e_bond = pallas_bonds_2body(
                positions, sys.bonds, sys.bond_params, sys.bond_mask,
                box=box_pbc, dist_eps=1e-10, interpret=interpret,
            )
            e_ub = pallas_bonds_2body(
                positions, sys.urey_bradley_bonds, sys.urey_bradley_params,
                sys.urey_bradley_mask, box=box_pbc, dist_eps=1e-10, interpret=interpret,
            )
            e_angle = pallas_angles_harmonic(
                positions, sys.angles, sys.angle_params, sys.angle_mask,
                box=box_pbc, dist_eps=1e-10, interpret=interpret,
            )
            e_dih = pallas_torsions_4body(
                positions, sys.dihedrals, sys.dihedral_params,
                sys.dihedral_mask, box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
            e_imp = pallas_torsions_4body(
                positions, sys.impropers, sys.improper_params,
                sys.improper_mask, box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
            )
            e_cmap = pallas_cmap_torsions(
                positions, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask,
                sys.cmap_coeffs, box=box_pbc, first_is_psi=True, interpret=interpret,
            )
            e_exc = pallas_exceptions_14(
                positions,
                getattr(sys, "exception_pairs", None),
                getattr(sys, "exception_sigmas", None),
                getattr(sys, "exception_epsilons", None),
                getattr(sys, "exception_chargeprods", None),
                getattr(sys, "exception_mask", None),
                box=box_pbc,
                soft_core_lambda=soft_core_lambda,
                softcore_alpha=float(getattr(sys, "softcore_alpha", 0.5)),
                softcore_exponent=int(getattr(sys, "softcore_exponent", 2)),
                interpret=interpret,
            )
        else:
            e_bond = _bond_energy_masked(
                positions, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn,
            )
            e_ub = _bond_energy_masked(
                positions, sys.urey_bradley_bonds, sys.urey_bradley_params,
                sys.urey_bradley_mask, displacement_fn,
            )
            e_angle = _angle_energy_masked(
                positions, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn,
            )
            e_dih = _dihedral_energy_masked(
                positions, sys.dihedrals, sys.dihedral_params,
                sys.dihedral_mask, displacement_fn,
            )
            e_imp = _dihedral_energy_masked(
                positions, sys.impropers, sys.improper_params,
                sys.improper_mask, displacement_fn,
            )
            e_cmap = _cmap_energy_masked(
                positions, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask,
                sys.cmap_coeffs, displacement_fn,
            )
            e_exc = _exception_energy_masked(
                positions,
                getattr(sys, "exception_pairs", None),
                getattr(sys, "exception_sigmas", None),
                getattr(sys, "exception_epsilons", None),
                getattr(sys, "exception_chargeprods", None),
                getattr(sys, "exception_mask", None),
                displacement_fn,
                soft_core_lambda=soft_core_lambda,
                softcore_alpha=float(getattr(sys, "softcore_alpha", 0.5)),
                softcore_exponent=int(getattr(sys, "softcore_exponent", 2)),
            )

        return e_bond + e_ub + e_angle + e_dih + e_imp + e_cmap + e_exc

    # Bonded + 1-4 exception forces via autodiff (safe path — no N² operations)
    bonded_force = -jax.grad(bonded_energy)(r)

    eff_elec_method = str(getattr(sys, "electrostatic_method", "pme")).lower()
    eff_lj_sw = float(getattr(sys, "lj_switch_width", 0.0))
    use_flash_eff = (
        use_flash
        and getattr(sys, "excl_indices", None) is not None
        and eff_elec_method not in ("reaction_field", "rf", "dsf", "damped_shifted_force", "cutoff")
        and eff_lj_sw <= 0.0
    )
    if use_flash_eff:
        # --- FlashMD path: O(N) memory, sparse exclusions ---
        if explicit_solvent:
            f_nonbonded = flash_explicit_forces(
                sys,
                T=flash_tile_size,
                soft_core_lambda=soft_core_lambda,
                remat=flash_remat,
                use_pallas=use_pallas,
            )
        else:
            f_nonbonded = flash_nonbonded_forces(
                sys,
                T=flash_tile_size,
                soft_core_lambda=soft_core_lambda,
            )
        total_force = (bonded_force + f_nonbonded) * sys.atom_mask[:, None]
    else:
        # --- Legacy path: dense (N,N) exclusion matrices required ---

        if sys.dense_excl_scale_vdw is not None:
            excl_scale_vdw = sys.dense_excl_scale_vdw
        else:
            excl_scale_vdw = jax.lax.stop_gradient(_build_dense_exclusion_scales(
                sys.excl_indices, sys.excl_scales_vdw, N,
            ))
        if sys.dense_excl_scale_elec is not None:
            excl_scale_elec = sys.dense_excl_scale_elec
        else:
            excl_scale_elec = jax.lax.stop_gradient(_build_dense_exclusion_scales(
                sys.excl_indices, sys.excl_scales_elec, N,
            ))

        if explicit_solvent or eff_elec_method in ("reaction_field", "rf", "dsf", "damped_shifted_force", "cutoff") or eff_lj_sw > 0.0:
            eff_cutoff = float(sys.nonbonded_cutoff)
            f_lj = -jax.grad(
                lambda pos: _lj_energy_masked(
                    pos, sys.sigmas, sys.epsilons, sys.atom_mask, displacement_fn,
                    soft_core_lambda=soft_core_lambda, excl_scale_vdw=excl_scale_vdw,
                    cutoff=eff_cutoff, lj_switch_width=eff_lj_sw,
                    softcore_alpha=float(getattr(sys, "softcore_alpha", 0.5)),
                    softcore_exponent=int(getattr(sys, "softcore_exponent", 2)),
                )
            )(r)
            f_coulomb = -jax.grad(
                lambda pos: _coulomb_energy_masked(
                    pos, sys.charges, sys.atom_mask, displacement_fn,
                    excl_scale_elec=excl_scale_elec,
                    pme_alpha=sys.pme_alpha if eff_elec_method == "pme" else 0.0,
                    electrostatic_method=eff_elec_method,
                    cutoff=eff_cutoff,
                    reaction_field_dielectric=float(getattr(sys, "reaction_field_dielectric", 78.3)),
                    dsf_alpha=float(getattr(sys, "dsf_alpha", 0.2)),
                )
            )(r)
        else:
            f_lj = lj_forces_dense(
                r, sys.sigmas, sys.epsilons, sys.atom_mask,
                soft_core_lambda=soft_core_lambda,
                excl_scale_vdw=excl_scale_vdw,
            )
            f_coulomb = coulomb_forces_dense(
                r, sys.charges, sys.atom_mask,
                excl_scale_elec=excl_scale_elec,
            )

        f_solv = jnp.zeros_like(r)
        if implicit_solvent:
            f_solv = gb_ace_forces_dense(
                r, sys.charges, sys.radii, sys.scaled_radii, sys.atom_mask,
                dielectric_offset=float(getattr(sys, "dielectric_offset", 0.09)),
                solvent_dielectric=float(getattr(sys, "solvent_dielectric", 78.5)),
                solute_dielectric=float(getattr(sys, "solute_dielectric", 1.0)),
            )

        total_force = bonded_force + f_lj + f_coulomb + f_solv

    if getattr(sys, "cosolvent_group_id", None) is not None and getattr(sys, "cosolvent_mol_id", None) is not None:
        f_cosolv_rep = -jax.grad(
            lambda pos: cosolvent_repulsive_energy(
                pos,
                sys.cosolvent_group_id,
                sys.cosolvent_mol_id,
                displacement_fn,
                eps_kcal=float(getattr(sys, "cosolvent_repulsive_eps", 0.01)),
                r0_angstrom=float(getattr(sys, "cosolvent_repulsive_r0", 10.0)),
                cutoff_angstrom=float(getattr(sys, "cosolvent_repulsive_cutoff", 9.0)),
                deduplicate_interaction_groups=bool(getattr(sys, "deduplicate_interaction_groups", False)),
                atom_mask=sys.atom_mask,
            )
        )(r)
        total_force = total_force + f_cosolv_rep

    # Belt-and-suspenders: zero forces on padding atoms.
    total_force = total_force * sys.atom_mask[:, None]
    return total_force.astype(r.dtype)

def make_batched_energy_fn(
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    use_pallas: bool | None = None,
) -> Callable[[PaddedSystem], Array]:
    """Create a vmap-compatible energy function for padded systems.
    
    Currently implements the full N^2 pairwise computation. Designed for batched
    execution of heterogeneous small systems.
    """
    return jax.vmap(lambda sys: single_padded_energy(sys, displacement_fn, implicit_solvent, use_pallas=use_pallas))


# ==============================================================================
# CUSTOM VJP LJ ENERGY — avoids scatter-add in backward pass
# ==============================================================================

def _make_lj_energy_nl_cvjp(neighbor_idx, soft_core_lambda=jnp.float32(1.0), excl_scales_vdw=None):
    """Factory creating a custom-VJP LJ energy function.

    Captures neighbor_idx and excl_scales_vdw via closure (non-differentiable).
    Returns a function f(r, sigmas, epsilons) → scalar energy.

    Key insight: with a symmetric NL and the 0.5 factor in the forward pass,
    the gradient w.r.t. r_i is simply the sum of pair-force vectors from
    row i of the NL. No scatter-add to other atoms is needed because the
    (j,i) pair in the NL already contributes the reverse force.

    This eliminates the expensive gather→scatter-add pattern that standard
    JAX autodiff generates for r[safe_idx] operations.
    """
    N_sentinel = neighbor_idx.shape[0]  # = N (padded atom count)

    def _compute_intermediates(r, sigmas, epsilons):
        """Shared forward computation for both energy and gradient."""
        safe_idx = jnp.minimum(neighbor_idx, N_sentinel - 1)
        r_j = r[safe_idx]                    # (N, K, 3)
        r_i = r[:, None, :]                  # (N, 1, 3)
        dr = r_i - r_j                       # (N, K, 3)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12)  # (N, K)

        sigma_ij = 0.5 * (sigmas[:, None] + sigmas[safe_idx])
        epsilon_ij = jnp.sqrt(epsilons[:, None] * epsilons[safe_idx])

        lam = jnp.float32(soft_core_lambda)
        alpha = jnp.float32(0.5)
        one_minus_lam = jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(0.0))
        soft_term = alpha * (one_minus_lam ** 2)
        r_over_sig = dist / jnp.maximum(sigma_ij, jnp.float32(1e-8))
        r6 = r_over_sig ** 6
        denom = soft_term + r6 + jnp.float32(1e-12)

        mask = neighbor_idx < N_sentinel  # (N, K)
        return dr, dist, sigma_ij, epsilon_ij, lam, denom, mask

    @jax.custom_vjp
    def lj_energy_cvjp(r, sigmas, epsilons):
        dr, dist, sigma_ij, epsilon_ij, lam, denom, mask = \
            _compute_intermediates(r, sigmas, epsilons)
        e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
            jnp.float32(1.0) / (denom * denom)
            - jnp.float32(1.0) / denom
        )
        e_pair = jnp.where(mask, e_pair, 0.0)
        # Apply exclusion scales (1-2/1-3 zeroed, 1-4 scaled)
        if excl_scales_vdw is not None:
            e_pair = e_pair * excl_scales_vdw
        return 0.5 * jnp.sum(e_pair)

    def _fwd(r, sigmas, epsilons):
        energy = lj_energy_cvjp(r, sigmas, epsilons)
        # Save minimal residuals for backward (recompute intermediates)
        return energy, (r, sigmas, epsilons)

    def _bwd(res, g):
        r, sigmas, epsilons = res
        dr, dist, sigma_ij, epsilon_ij, lam, denom, mask = \
            _compute_intermediates(r, sigmas, epsilons)

        # Analytical derivative: de_pair/d(dist)
        sigma_ij6 = jnp.maximum(sigma_ij ** 6, jnp.float32(1e-48))
        ddenom_ddist = jnp.float32(6.0) * dist ** 5 / sigma_ij6
        de_ddist = jnp.float32(4.0) * epsilon_ij * lam * (
            jnp.float32(1.0) / (denom ** 2)
            - jnp.float32(2.0) / (denom ** 3)
        ) * ddenom_ddist

        de_ddist = jnp.where(mask, de_ddist, 0.0)
        # Apply exclusion scales to gradient too
        if excl_scales_vdw is not None:
            de_ddist = de_ddist * excl_scales_vdw

        # Gradient: de/d(r_i) = de/d(dist) · (r_i - r_j) / dist
        unit_dr = dr / (dist[..., None] + 1e-12)  # (N, K, 3)
        pair_grad = de_ddist[..., None] * unit_dr   # (N, K, 3)
        grad_r = jnp.sum(pair_grad, axis=1)         # (N, 3)

        return (g * grad_r, jnp.zeros_like(sigmas), jnp.zeros_like(epsilons))

    lj_energy_cvjp.defvjp(_fwd, _bwd)
    return lj_energy_cvjp


def _make_coulomb_energy_nl_cvjp(neighbor_idx, excl_scales_elec=None):
    """Factory creating a custom-VJP O(N*K) Coulomb energy function over a symmetric neighbor list."""
    N_sentinel = neighbor_idx.shape[0]
    i_idx = jnp.arange(N_sentinel, dtype=neighbor_idx.dtype)[:, None]

    def _compute_intermediates(r, charges):
        safe_idx = jnp.minimum(neighbor_idx, N_sentinel - 1)
        r_j = r[safe_idx]
        r_i = r[:, None, :]
        dr = r_i - r_j
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))
        dist_safe = jnp.where(dist < 1e-4, 1.0, dist)
        q_i = charges[:, None]
        q_j = charges[safe_idx]
        mask = (neighbor_idx >= 0) & (neighbor_idx < N_sentinel) & (neighbor_idx != i_idx) & (dist >= 1e-4)
        return dr, dist_safe, q_i, q_j, mask

    @jax.custom_vjp
    def coulomb_energy_cvjp(r, charges):
        _, dist_safe, q_i, q_j, mask = _compute_intermediates(r, charges)
        e_pair = COULOMB_CONSTANT * (q_i * q_j) / dist_safe
        e_pair = jnp.where(mask, e_pair, 0.0)
        if excl_scales_elec is not None:
            e_pair = e_pair * excl_scales_elec
        return 0.5 * jnp.sum(e_pair)

    def _fwd(r, charges):
        return coulomb_energy_cvjp(r, charges), (r, charges)

    def _bwd(res, g):
        r, charges = res
        dr, dist_safe, q_i, q_j, mask = _compute_intermediates(r, charges)
        scale = excl_scales_elec if excl_scales_elec is not None else 1.0
        de_dr_over_r = jnp.where(
            mask,
            -COULOMB_CONSTANT * (q_i * q_j) * scale / (dist_safe ** 3),
            0.0,
        )
        grad_r = jnp.sum(de_dr_over_r[..., None] * dr, axis=1)
        grad_q = jnp.sum(
            jnp.where(mask, COULOMB_CONSTANT * q_j * scale / dist_safe, 0.0),
            axis=1,
        )
        return (g * grad_r, g * grad_q)

    coulomb_energy_cvjp.defvjp(_fwd, _bwd)
    return coulomb_energy_cvjp


def single_padded_energy_nl_cvjp(
    sys: PaddedSystem,
    neighbor_idx: Array,
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    soft_core_lambda: Array = jnp.array(1.0),
    use_pallas: bool | None = None,
) -> Array:
    """Like single_padded_energy_nl but with custom VJP for LJ/Coulomb + checkpoint for GB.

    This variant:
    1. Uses analytical LJ and Coulomb forces via custom_vjp in O(N*K) (no scatter-add, no O(N^2))
    2. Wraps GB in jax.checkpoint (recompute forward during backward)

    Together this minimizes both compute and memory in the gradient pass.
    """
    r = sys.positions
    pallas_enabled, interpret = resolve_pallas_mode(use_pallas=use_pallas)
    box_pbc = _box_for_displacement_fn(sys.box_size, displacement_fn)
    _has_pallas_bonded = (
        pallas_bonds_2body is not None
        and pallas_angles_harmonic is not None
        and pallas_torsions_4body is not None
        and pallas_cmap_torsions is not None
    )

    # Bonded terms (unchanged — already O(N))
    if pallas_enabled and _has_pallas_bonded:
        e_bond = pallas_bonds_2body(
            r, sys.bonds, sys.bond_params, sys.bond_mask,
            box=box_pbc, dist_eps=1e-10, interpret=interpret,
        )
        e_ub = pallas_bonds_2body(
            r, sys.urey_bradley_bonds, sys.urey_bradley_params, sys.urey_bradley_mask,
            box=box_pbc, dist_eps=1e-10, interpret=interpret,
        )
        e_angle = pallas_angles_harmonic(
            r, sys.angles, sys.angle_params, sys.angle_mask,
            box=box_pbc, dist_eps=1e-10, interpret=interpret,
        )
        e_dih = pallas_torsions_4body(
            r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask,
            box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
        )
        e_imp = pallas_torsions_4body(
            r, sys.impropers, sys.improper_params, sys.improper_mask,
            box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
        )
        e_cmap = pallas_cmap_torsions(
            r, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs,
            box=box_pbc, first_is_psi=True, interpret=interpret,
        )
    else:
        e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
        e_ub = _bond_energy_masked(r, sys.urey_bradley_bonds, sys.urey_bradley_params, sys.urey_bradley_mask, displacement_fn)
        e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
        e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
        e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
        e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)

    N = len(sys.atom_mask)
    nb_idx = jnp.where(
        sys.atom_mask[:, None] & sys.atom_mask[jnp.minimum(neighbor_idx, N - 1)],
        neighbor_idx,
        N,
    )

    if pallas_enabled and pallas_nonbonded_nl is not None:
        e_lj = pallas_nonbonded_nl(
            r,
            sys.charges,
            sys.sigmas,
            sys.epsilons,
            sys.atom_mask,
            nb_idx,
            sys.excl_indices,
            sys.excl_scales_vdw,
            sys.excl_scales_elec,
            box=None,
            pme_alpha=0.0,
            cutoff=1e9,
            lj_switch_width=0.0,
            soft_core_lambda=soft_core_lambda,
            include_pme_excl_erf=False,
            interpret=interpret,
        )
        e_elec = jnp.asarray(0.0, dtype=r.dtype)
    else:
        # Compute (N, K) exclusion scales for neighbor list
        excl_scales_vdw_nl, excl_scales_elec_nl = get_neighbor_exclusion_scales(
            sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, nb_idx,
        )

        # LJ with custom VJP (analytical forces, no scatter-add)
        lj_fn = _make_lj_energy_nl_cvjp(
            nb_idx, soft_core_lambda, excl_scales_vdw=excl_scales_vdw_nl,
        )
        e_lj = lj_fn(r, sys.sigmas, sys.epsilons)

        # O(N*K) neighbor-list Coulomb with custom VJP
        coul_fn = _make_coulomb_energy_nl_cvjp(
            nb_idx, excl_scales_elec=excl_scales_elec_nl,
        )
        e_elec = coul_fn(r, sys.charges)

    if implicit_solvent:
        if pallas_enabled and pallas_gb_ace_nl is not None:
            e_solv = pallas_gb_ace_nl(
                r,
                sys.charges,
                sys.radii,
                nb_idx,
                atom_mask=sys.atom_mask,
                scaled_radii=sys.scaled_radii,
                dielectric_offset=0.09,
                solvent_dielectric=78.5,
                solute_dielectric=1.0,
                surface_tension=0.0054,
                box=None,
                interpret=interpret,
            )
        else:
            # Checkpoint GB to save memory (recompute forward during backward)
            @jax.checkpoint
            def _gb_energy(positions, charges, radii):
                e_gb, born_radii = compute_gb_energy_neighbor_list(
                    positions=positions,
                    charges=charges,
                    radii=radii,
                    neighbor_idx=nb_idx,
                    dielectric_offset=0.09,
                    scaled_radii=sys.scaled_radii,
                )
                e_np = compute_ace_nonpolar_energy(radii, born_radii)
                e_np = jnp.sum(e_np * sys.atom_mask)
                return e_gb + e_np

            e_solv = _gb_energy(r, sys.charges, sys.radii)
    else:
        e_solv = 0.0

    e_cosolv_rep = jnp.float32(0.0)
    if getattr(sys, "cosolvent_group_id", None) is not None and getattr(sys, "cosolvent_mol_id", None) is not None:
        e_cosolv_rep = cosolvent_repulsive_energy(
            r,
            sys.cosolvent_group_id,
            sys.cosolvent_mol_id,
            displacement_fn,
            eps_kcal=float(getattr(sys, "cosolvent_repulsive_eps", 0.01)),
            r0_angstrom=float(getattr(sys, "cosolvent_repulsive_r0", 10.0)),
            cutoff_angstrom=float(getattr(sys, "cosolvent_repulsive_cutoff", 9.0)),
            deduplicate_interaction_groups=bool(getattr(sys, "deduplicate_interaction_groups", False)),
            atom_mask=sys.atom_mask,
        )

    if pallas_enabled and pallas_exceptions_14 is not None:
        e_exc = pallas_exceptions_14(
            r,
            getattr(sys, "exception_pairs", None),
            getattr(sys, "exception_sigmas", None),
            getattr(sys, "exception_epsilons", None),
            getattr(sys, "exception_chargeprods", None),
            getattr(sys, "exception_mask", None),
            box=box_pbc,
            soft_core_lambda=1.0,
            interpret=interpret,
        )
    else:
        e_exc = _exception_energy_masked(
            r,
            getattr(sys, "exception_pairs", None),
            getattr(sys, "exception_sigmas", None),
            getattr(sys, "exception_epsilons", None),
            getattr(sys, "exception_chargeprods", None),
            getattr(sys, "exception_mask", None),
            displacement_fn,
        )

    return e_bond + e_ub + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv + e_cosolv_rep + e_exc


def single_padded_energy_nl(
    sys: PaddedSystem,
    neighbor_idx: Array,
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    soft_core_lambda: Array = jnp.array(1.0),
    use_pallas: bool | None = None,
) -> Array:
    """Computes total potential energy using neighbor lists for non-bonded terms.

    O(N*K) scaling for LJ, Coulomb, and GB instead of O(N^2).
    Bonded terms remain identical to single_padded_energy.

    Args:
        sys: Padded protein system.
        neighbor_idx: Neighbor indices (N, K). Padding sentinel = N.
        displacement_fn: JAX-MD displacement function.
        implicit_solvent: Whether to use GB/SA implicit solvent.
        soft_core_lambda: Soft-core LJ coupling (1.0 = standard).

    Returns:
        Total potential energy (scalar).
    """
    r = sys.positions
    pallas_enabled, interpret = resolve_pallas_mode(use_pallas=use_pallas)
    box_pbc = _box_for_displacement_fn(sys.box_size, displacement_fn)
    _has_pallas_bonded = (
        pallas_bonds_2body is not None
        and pallas_angles_harmonic is not None
        and pallas_torsions_4body is not None
        and pallas_cmap_torsions is not None
    )

    # Bonded terms (unchanged — already O(N))
    if pallas_enabled and _has_pallas_bonded:
        e_bond = pallas_bonds_2body(
            r, sys.bonds, sys.bond_params, sys.bond_mask,
            box=box_pbc, dist_eps=1e-10, interpret=interpret,
        )
        e_ub = pallas_bonds_2body(
            r, sys.urey_bradley_bonds, sys.urey_bradley_params, sys.urey_bradley_mask,
            box=box_pbc, dist_eps=1e-10, interpret=interpret,
        )
        e_angle = pallas_angles_harmonic(
            r, sys.angles, sys.angle_params, sys.angle_mask,
            box=box_pbc, dist_eps=1e-10, interpret=interpret,
        )
        e_dih = pallas_torsions_4body(
            r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask,
            box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
        )
        e_imp = pallas_torsions_4body(
            r, sys.impropers, sys.improper_params, sys.improper_mask,
            box=box_pbc, b0_is_ji=True, is_harmonic=False, interpret=interpret,
        )
        e_cmap = pallas_cmap_torsions(
            r, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs,
            box=box_pbc, first_is_psi=True, interpret=interpret,
        )
    else:
        e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
        e_ub = _bond_energy_masked(r, sys.urey_bradley_bonds, sys.urey_bradley_params, sys.urey_bradley_mask, displacement_fn)
        e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
        e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
        e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
        e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)

    N = len(sys.atom_mask)
    nb_idx = jnp.where(
        sys.atom_mask[:, None] & sys.atom_mask[jnp.minimum(neighbor_idx, N - 1)],
        neighbor_idx,
        N,
    )

    if pallas_enabled and pallas_nonbonded_nl is not None:
        e_lj = pallas_nonbonded_nl(
            r,
            sys.charges,
            sys.sigmas,
            sys.epsilons,
            sys.atom_mask,
            nb_idx,
            sys.excl_indices,
            sys.excl_scales_vdw,
            sys.excl_scales_elec,
            box=None,
            pme_alpha=0.0,
            cutoff=1e9,
            lj_switch_width=0.0,
            soft_core_lambda=soft_core_lambda,
            include_pme_excl_erf=False,
            interpret=interpret,
        )
        e_elec = jnp.asarray(0.0, dtype=r.dtype)
    else:
        # Non-bonded: O(N*K) via neighbor list
        excl_scales_vdw_nl, excl_scales_elec_nl = get_neighbor_exclusion_scales(
            sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, nb_idx,
        )

        e_lj = _lj_energy_neighbor_list(
            r, sys.sigmas, sys.epsilons, nb_idx,
            soft_core_lambda=soft_core_lambda,
            excl_scales_vdw=excl_scales_vdw_nl,
        )

        e_elec = _coulomb_energy_neighbor_list(
            r, sys.charges, nb_idx,
            excl_scales_elec=excl_scales_elec_nl,
        )

    if implicit_solvent:
        if pallas_enabled and pallas_gb_ace_nl is not None:
            e_solv = pallas_gb_ace_nl(
                r,
                sys.charges,
                sys.radii,
                nb_idx,
                atom_mask=sys.atom_mask,
                scaled_radii=sys.scaled_radii,
                dielectric_offset=0.09,
                solvent_dielectric=78.5,
                solute_dielectric=1.0,
                surface_tension=0.0054,
                box=None,
                interpret=interpret,
            )
        else:
            e_gb, born_radii = compute_gb_energy_neighbor_list(
                positions=r,
                charges=sys.charges,
                radii=sys.radii,
                neighbor_idx=nb_idx,
                dielectric_offset=0.09,
                scaled_radii=sys.scaled_radii,
            )
            e_np = compute_ace_nonpolar_energy(sys.radii, born_radii)
            e_np = jnp.sum(e_np * sys.atom_mask)
            e_solv = e_gb + e_np
    else:
        e_solv = 0.0

    e_cosolv_rep = jnp.float32(0.0)
    if getattr(sys, "cosolvent_group_id", None) is not None and getattr(sys, "cosolvent_mol_id", None) is not None:
        e_cosolv_rep = cosolvent_repulsive_energy(
            r,
            sys.cosolvent_group_id,
            sys.cosolvent_mol_id,
            displacement_fn,
            eps_kcal=float(getattr(sys, "cosolvent_repulsive_eps", 0.01)),
            r0_angstrom=float(getattr(sys, "cosolvent_repulsive_r0", 10.0)),
            cutoff_angstrom=float(getattr(sys, "cosolvent_repulsive_cutoff", 9.0)),
            deduplicate_interaction_groups=bool(getattr(sys, "deduplicate_interaction_groups", False)),
            atom_mask=sys.atom_mask,
        )

    if pallas_enabled and pallas_exceptions_14 is not None:
        e_exc = pallas_exceptions_14(
            r,
            getattr(sys, "exception_pairs", None),
            getattr(sys, "exception_sigmas", None),
            getattr(sys, "exception_epsilons", None),
            getattr(sys, "exception_chargeprods", None),
            getattr(sys, "exception_mask", None),
            box=box_pbc,
            soft_core_lambda=1.0,
            interpret=interpret,
        )
    else:
        e_exc = _exception_energy_masked(
            r,
            getattr(sys, "exception_pairs", None),
            getattr(sys, "exception_sigmas", None),
            getattr(sys, "exception_epsilons", None),
            getattr(sys, "exception_chargeprods", None),
            getattr(sys, "exception_mask", None),
            displacement_fn,
        )

    return e_bond + e_ub + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv + e_cosolv_rep + e_exc


def make_batched_energy_fn_nl(
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    use_pallas: bool | None = None,
) -> Callable:
    """Create a vmap-compatible energy function using neighbor lists.

    Unlike make_batched_energy_fn, this takes neighbor_idx as a separate
    argument (shape [B, N, K]) since it's updated dynamically during simulation.

    Returns:
        Function(sys_batch: PaddedSystem, neighbor_idx: Array) -> Array[B]
    """
    def _energy_single(sys, nbr_idx):
        return single_padded_energy_nl(
            sys, nbr_idx, displacement_fn, implicit_solvent, use_pallas=use_pallas,
        )

    return jax.vmap(_energy_single)
