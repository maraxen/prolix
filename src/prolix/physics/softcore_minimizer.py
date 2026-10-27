"""Staged Beutler soft-core minimization and OpenMM-convention positional restraints.

Implements `FR-8` (`CA-2`, `R-1`, `NBB-2`) for `naurmalade` parity:
- Beutler (1994) soft-core Lennard-Jones potential with configurable exponent
  ``p = 2`` (`(1 - lambda_lj)^2`, matching `naurmalade` `SoftcoreSweepStep`):
  ``s = alpha_sc * (1 - lambda_lj)^p + (r / sigma)^6``,
  ``U_LJ,sc(r; lambda_lj) = 4 * epsilon * lambda_lj * (1 / s^2 - 1 / s)``.
- Optional soft-core cutoff Coulomb potential with ``alpha_q = 0.01 A^2``
  (`0.0001 nm^2`), defaulting to ``lambda_elec = 0.0`` when ``soft_coulomb=False``.
- Positional restraints using OpenMM's unhalved ``CustomExternalForce`` convention
  (``U_rest = s_rest * sum_i m_i * k_rest * (Delta d_i)^2`` with
  ``k_rest = k_kj_nm2 / 418.4 kcal/(mol*A^2)``) and periodic minimum-image
  displacements.
- Staged soft-core window sweep with PME reciprocal space, PME exclusion
  corrections, and isotropic LJ dispersion tail disabled during soft-core windows,
  followed by full-Hamiltonian cleanup at ``(lambda_lj=1.0, lambda_elec=1.0)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax_md import space
from proxide.physics.constants import COULOMB_CONSTANT

from prolix.physics import bonded

if TYPE_CHECKING:
    from jax_md.util import Array

KJ_NM2_TO_KCAL_A2_DIVISOR: float = 418.4

SOLVENT_AND_ION_RESIDUES: frozenset[str] = frozenset(
    {"HOH", "WAT", "TIP3", "NA", "CL", "K", "MG", "ZN"}
)

BACKBONE_ATOM_NAMES: frozenset[str] = frozenset({"N", "CA", "C", "O"})
CA_ATOM_NAMES: frozenset[str] = frozenset({"CA"})


@dataclass(frozen=True)
class SoftcoreWindow:
    """Single stage in a soft-core minimization lambda schedule.

    Attributes:
        lambda_lj: Soft-core LJ coupling parameter in ``(0, 1]``.
        lambda_elec: Electrostatic coupling parameter in ``[0, 1]``. Defaults to
            ``0.0`` (`naurmalade` disables electrostatics during the soft-core
            LJ sweep unless ``soft_coulomb=True``).
        restraint_scale: Multiplicative scale factor on positional restraints
            during this stage.
        max_iterations: Maximum optimizer iterations in this window.
    """

    lambda_lj: float
    lambda_elec: float = 0.0
    restraint_scale: float = 1.0
    max_iterations: int = 100


DEFAULT_NAURMALADE_WINDOWS: tuple[SoftcoreWindow, ...] = (
    SoftcoreWindow(lambda_lj=0.5, lambda_elec=0.0, restraint_scale=1.0, max_iterations=100),
    SoftcoreWindow(lambda_lj=0.7, lambda_elec=0.0, restraint_scale=1.0, max_iterations=100),
    SoftcoreWindow(lambda_lj=0.85, lambda_elec=0.0, restraint_scale=1.0, max_iterations=100),
    SoftcoreWindow(lambda_lj=0.95, lambda_elec=0.0, restraint_scale=1.0, max_iterations=100),
    SoftcoreWindow(lambda_lj=1.0, lambda_elec=1.0, restraint_scale=1.0, max_iterations=100),
)


def k_kj_nm2_to_kcal_a2(k_kj_nm2: float) -> float:
    """Convert OpenMM restraint force constant from ``kJ/(mol*nm^2)`` to ``kcal/(mol*A^2)``.

    ``1 kJ/mol = 1 / 4.184 kcal/mol`` and ``1 nm^2 = 100 A^2``, so
    ``k_kcal_a2 = k_kj_nm2 / 418.4``.
    """
    return float(k_kj_nm2) / KJ_NM2_TO_KCAL_A2_DIVISOR


def _resolve_pair_distances_sq_and_params(
    r: Array,
    pairs_or_positions: Array | None,
    sigmas: Array,
    epsilons: Array,
    mask: Array | None,
    displacement_fn: Callable[[Array, Array], Array] | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Resolve squared pair distances, pair `(sigma, epsilon)`, pair mask, and symmetry scale.

    Supports three input modes:
    1. Precomputed scalar/1D distances ``r`` (when ``pairs_or_positions is None`` and
       ``r.ndim <= 1`` or ``displacement_fn is None``).
    2. Indexed pair list ``pairs_or_positions`` of integer dtype and shape ``(P, 2)``
       with atomic positions ``r`` of shape ``(N, 3)``.
    3. Dense all-pairs positions ``r`` of shape ``(N, 3)`` (with ``pairs_or_positions``
       either ``None`` or a second ``(N, 3)`` coordinate array).
    """
    r_arr = jnp.asarray(r)
    sig_arr = jnp.asarray(sigmas, dtype=r_arr.dtype)
    eps_arr = jnp.asarray(epsilons, dtype=r_arr.dtype)

    if pairs_or_positions is not None:
        pop_arr = jnp.asarray(pairs_or_positions)
        if jnp.issubdtype(pop_arr.dtype, jnp.integer) and pop_arr.ndim == 2 and pop_arr.shape[-1] == 2:
            if displacement_fn is None:
                raise ValueError("displacement_fn is required when pairs_or_positions is a (P, 2) index array.")
            n_atoms = r_arr.shape[0]
            p_count = pop_arr.shape[0]
            if mask is None:
                pair_mask = jnp.ones((p_count,), dtype=r_arr.dtype)
            else:
                pair_mask = jnp.asarray(mask, dtype=r_arr.dtype)
            i_idx = jnp.clip(pop_arr[:, 0], 0, max(n_atoms - 1, 0))
            j_idx = jnp.clip(pop_arr[:, 1], 0, max(n_atoms - 1, 0))
            dr = jax.vmap(displacement_fn)(r_arr[i_idx], r_arr[j_idx])
            dist_sq = jnp.sum(dr * dr, axis=-1)
            if sig_arr.ndim == 1 and sig_arr.shape[0] == p_count and (
                n_atoms != p_count or eps_arr.shape[0] == p_count
            ):
                # Check if per-pair or per-atom parameters were supplied
                sigma_ij = sig_arr
                epsilon_ij = eps_arr
            else:
                sigma_ij = 0.5 * (sig_arr[i_idx] + sig_arr[j_idx])
                epsilon_ij = jnp.sqrt(jnp.maximum(eps_arr[i_idx] * eps_arr[j_idx], 0.0))
            return dist_sq, sigma_ij, epsilon_ij, pair_mask, jnp.asarray(1.0, dtype=r_arr.dtype)

    if r_arr.ndim <= 1 or displacement_fn is None:
        dist_sq = jnp.square(r_arr)
        if mask is None:
            pair_mask = jnp.ones_like(dist_sq, dtype=r_arr.dtype)
        else:
            pair_mask = jnp.asarray(mask, dtype=r_arr.dtype)
        return dist_sq, sig_arr, eps_arr, pair_mask, jnp.asarray(1.0, dtype=r_arr.dtype)

    # Dense (N, 3) positions
    r_other = r_arr if pairs_or_positions is None else jnp.asarray(pairs_or_positions, dtype=r_arr.dtype)
    dr = space.map_product(displacement_fn)(r_arr, r_other)
    dist_sq = jnp.sum(dr * dr, axis=-1)
    n_atoms = r_arr.shape[0]
    if sig_arr.ndim == 1:
        sigma_ij = 0.5 * (sig_arr[:, None] + sig_arr[None, :])
        epsilon_ij = jnp.sqrt(jnp.maximum(eps_arr[:, None] * eps_arr[None, :], 0.0))
    else:
        sigma_ij = sig_arr
        epsilon_ij = eps_arr

    if mask is None:
        pair_mask = 1.0 - jnp.eye(n_atoms, dtype=r_arr.dtype)
    else:
        mask_arr = jnp.asarray(mask)
        if mask_arr.ndim == 1:
            mask_2d = (mask_arr[:, None] > 0) & (mask_arr[None, :] > 0)
            pair_mask = mask_2d.astype(r_arr.dtype) * (1.0 - jnp.eye(n_atoms, dtype=r_arr.dtype))
        else:
            pair_mask = mask_arr.astype(r_arr.dtype) * (1.0 - jnp.eye(n_atoms, dtype=r_arr.dtype))

    symmetry_scale = jnp.asarray(0.5 if pairs_or_positions is None else 1.0, dtype=r_arr.dtype)
    return dist_sq, sigma_ij, epsilon_ij, pair_mask, symmetry_scale


def compute_beutler_softcore_lj_energy(
    r: Array,
    pairs_or_positions: Array | None,
    sigmas: Array,
    epsilons: Array,
    mask: Array | None,
    displacement_fn: Callable[[Array, Array], Array] | None = None,
    *,
    lambda_lj: float | Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
) -> Array:
    """Compute Beutler (1994) soft-core Lennard-Jones energy (`CA-2` / `R-1`).

    Formula:
        ``s = alpha_sc * (1 - lambda_lj)^p + (r / sigma)^6``
        ``U_LJ,sc(r; lambda_lj) = 4 * epsilon * lambda_lj * (1 / s^2 - 1 / s)``

    Uses ``(r^2 / sigma^2)^3`` directly from squared displacements so both the
    potential and its analytical gradient w.r.t. coordinates are smooth and
    finite even at ``r = 0`` when ``lambda_lj < 1``.

    Args:
        r: Either precomputed pair distances ``(...,)`` or atomic coordinates ``(N, 3)``.
        pairs_or_positions: Optional ``(P, 2)`` pair index array, second ``(N, 3)``
            coordinate array, or ``None`` (for precomputed distances or self-dense ``(N, 3)``).
        sigmas: Per-pair or per-atom LJ ``sigma`` in Angstroms.
        epsilons: Per-pair or per-atom LJ ``epsilon`` in kcal/mol.
        mask: Boolean or float mask (per-pair ``(P,)``, per-atom ``(N,)``, or dense ``(N, N)``).
        displacement_fn: Periodic or free-space displacement function ``(r_i, r_j) -> dr``.
        lambda_lj: Soft-core coupling parameter in ``[0, 1]``.
        softcore_alpha: Soft-core shift parameter ``alpha_sc`` (default ``0.5``).
        softcore_exponent: Exponent ``p`` on ``(1 - lambda_lj)^p``. Default ``2``
            matches `naurmalade` (`(1 - lambda_lj)^2`); ``1`` gives linear `(1 - lambda_lj)`.

    Returns:
        Scalar soft-core Lennard-Jones energy in kcal/mol.
    """
    if softcore_exponent not in (1, 2):
        raise ValueError(f"softcore_exponent must be 1 or 2, got {softcore_exponent}")

    dist_sq, sigma_ij, epsilon_ij, pair_mask, symmetry_scale = _resolve_pair_distances_sq_and_params(
        r, pairs_or_positions, sigmas, epsilons, mask, displacement_fn
    )
    dtype = dist_sq.dtype
    lam = jnp.asarray(lambda_lj, dtype=dtype)
    alpha = jnp.asarray(softcore_alpha, dtype=dtype)

    mask_bool = pair_mask > 0
    sigma_safe = jnp.where(mask_bool, jnp.maximum(sigma_ij, jnp.asarray(1e-4, dtype=dtype)), jnp.ones_like(sigma_ij))
    one_minus_lam = jnp.maximum(jnp.asarray(1.0, dtype=dtype) - lam, jnp.asarray(0.0, dtype=dtype))
    soft_term = alpha * (one_minus_lam ** softcore_exponent)

    r2_over_sig2 = jnp.where(mask_bool, dist_sq / (sigma_safe * sigma_safe), jnp.ones_like(dist_sq))
    r6 = r2_over_sig2 * r2_over_sig2 * r2_over_sig2

    # When lambda_lj == 1.0 and r == 0 on masked-out pairs, r2_over_sig2 is 1.0 so s >= 1.0.
    # For active pairs at lambda_lj < 1.0, soft_term > 0 keeps s strictly positive even at r = 0.
    s = soft_term + r6
    s_safe = jnp.where(mask_bool, jnp.maximum(s, jnp.asarray(1e-12, dtype=dtype)), jnp.ones_like(s))

    e_pair = jnp.asarray(4.0, dtype=dtype) * epsilon_ij * lam * (
        jnp.asarray(1.0, dtype=dtype) / (s_safe * s_safe) - jnp.asarray(1.0, dtype=dtype) / s_safe
    )
    e_masked = jnp.where(mask_bool, e_pair * pair_mask, jnp.zeros_like(e_pair))
    return symmetry_scale * jnp.sum(e_masked)


def compute_softcore_coulomb_energy(
    r: Array,
    pairs_or_positions: Array | None,
    charges_or_chargeprods: Array,
    mask: Array | None,
    displacement_fn: Callable[[Array, Array], Array] | None = None,
    *,
    lambda_elec: float | Array = 0.0,
    soft_coulomb: bool = False,
    alpha_q_angstrom2: float = 0.01,
    coulomb_constant: float = COULOMB_CONSTANT,
    cutoff: float | None = None,
    softcore_coulomb_exponent: int = 2,
) -> Array:
    """Compute soft-core cutoff Coulomb energy for `SoftcoreSweepStep` (`FR-8`).

    When ``soft_coulomb=False`` (`naurmalade` default during `SoftcoreSweepStep`),
    electrostatics are disabled (``lambda_elec = 0.0``) and this function returns 0.
    When ``soft_coulomb=True``, evaluates:
        ``U_elec,sc(r; lambda_elec) = lambda_elec * C * q_i * q_j / sqrt(r^2 + alpha_q * (1 - lambda_elec)^2)``
    with ``alpha_q = 0.01 A^2`` (`100 * 0.0001 nm^2`).
    """
    r_arr = jnp.asarray(r)
    if not soft_coulomb:
        return jnp.zeros((), dtype=r_arr.dtype)

    q_arr = jnp.asarray(charges_or_chargeprods, dtype=r_arr.dtype)
    dummy_eps = jnp.ones_like(q_arr)
    dist_sq, _, _, pair_mask, symmetry_scale = _resolve_pair_distances_sq_and_params(
        r_arr, pairs_or_positions, q_arr, dummy_eps, mask, displacement_fn
    )
    dtype = dist_sq.dtype

    # Resolve charge product q_i * q_j
    if pairs_or_positions is not None:
        pop_arr = jnp.asarray(pairs_or_positions)
        if jnp.issubdtype(pop_arr.dtype, jnp.integer) and pop_arr.ndim == 2 and pop_arr.shape[-1] == 2:
            n_atoms = r_arr.shape[0]
            p_count = pop_arr.shape[0]
            i_idx = jnp.clip(pop_arr[:, 0], 0, max(n_atoms - 1, 0))
            j_idx = jnp.clip(pop_arr[:, 1], 0, max(n_atoms - 1, 0))
            if q_arr.ndim == 1 and q_arr.shape[0] == p_count and n_atoms != p_count:
                q_prod = q_arr
            else:
                q_prod = q_arr[i_idx] * q_arr[j_idx]
        else:
            q_prod = q_arr[:, None] * q_arr[None, :] if q_arr.ndim == 1 else q_arr
    elif r_arr.ndim <= 1 or displacement_fn is None:
        q_prod = q_arr
    else:
        q_prod = q_arr[:, None] * q_arr[None, :] if q_arr.ndim == 1 else q_arr

    if cutoff is not None:
        pair_mask = pair_mask * (dist_sq < (float(cutoff) ** 2)).astype(dtype)

    lam_e = jnp.asarray(lambda_elec, dtype=dtype)
    alpha_q = jnp.asarray(alpha_q_angstrom2, dtype=dtype)
    one_minus_lam = jnp.maximum(jnp.asarray(1.0, dtype=dtype) - lam_e, jnp.asarray(0.0, dtype=dtype))
    soft_q = alpha_q * (one_minus_lam ** int(softcore_coulomb_exponent))

    mask_bool = pair_mask > 0
    denom_sq = jnp.where(
        mask_bool,
        jnp.maximum(dist_sq + soft_q, jnp.asarray(1e-12, dtype=dtype)),
        jnp.ones_like(dist_sq),
    )
    inv_r_sc = jax.lax.rsqrt(denom_sq)
    e_pair = lam_e * jnp.asarray(coulomb_constant, dtype=dtype) * q_prod * inv_r_sc
    e_masked = jnp.where(mask_bool, e_pair * pair_mask, jnp.zeros_like(e_pair))
    return symmetry_scale * jnp.sum(e_masked)


def compute_positional_restraint_energy(
    positions: Array,
    reference_positions: Array,
    restraint_mask: Array,
    displacement_fn: Callable[[Array, Array], Array],
    *,
    k_kcal_mol_a2: float | Array = 0.0,
    flat_bottom_radius_angstrom: float = 0.0,
    restraint_scale: float | Array = 1.0,
) -> Array:
    """Compute OpenMM-convention unhalved positional restraint energy (`CA-2` / `NBB-2`).

    Matches `naurmalade/minimization/restraints.py` (`CustomExternalForce`
    ``"k_rest * periodicdistance(x, y, z, x0, y0, z0)^2"`` without a ``1/2`` factor):
        ``U_rest(r) = s_rest * sum_i m_i * k_rest * (Delta d_i)^2``
    where ``dr_i = displacement_fn(r_i, r_{0,i})``,
    ``d_i = sqrt(||dr_i||^2 + 1e-12)``, and ``Delta d_i = max(0, d_i - r_fb)``.
     Consequently, for ``r_fb = 0``, the restoring force is ``F_i = -2 * s_rest * k_rest * dr_i``.
    """
    pos = jnp.asarray(positions)
    ref = jnp.asarray(reference_positions, dtype=pos.dtype)
    mask = jnp.asarray(restraint_mask, dtype=pos.dtype)
    dtype = pos.dtype

    k_val = jnp.asarray(k_kcal_mol_a2, dtype=dtype)
    scale_val = jnp.asarray(restraint_scale, dtype=dtype)
    r_fb = jnp.asarray(flat_bottom_radius_angstrom, dtype=dtype)

    dr = jax.vmap(displacement_fn)(pos, ref)
    dist_sq = jnp.sum(dr * dr, axis=-1)
    dist = jnp.sqrt(dist_sq + jnp.asarray(1e-12, dtype=dtype))
    delta_d = jnp.maximum(jnp.asarray(0.0, dtype=dtype), dist - r_fb)
    excess_sq = jnp.where(r_fb > jnp.asarray(0.0, dtype=dtype), delta_d * delta_d, dist_sq)

    return scale_val * k_val * jnp.sum(mask * excess_sq)


def _is_heavy_atom_name(atom_name: str) -> bool:
    """Return True if ``atom_name`` is a non-hydrogen heavy atom."""
    stripped = atom_name.strip().upper()
    if not stripped:
        return False
    if stripped[0] == "H":
        return False
    if len(stripped) >= 2 and stripped[0].isdigit() and stripped[1] == "H":
        return False
    return True


def select_restraint_mask(
    atom_names: Sequence[str],
    residue_names: Sequence[str] | None = None,
    *,
    selection: str = "backbone",
    atom_mask: Array | None = None,
    exclude_residue_names: Iterable[str] | None = None,
) -> Array:
    """Build a boolean/float restraint mask over atoms (`FR-8`).

    Supported ``selection`` values (case-insensitive):
    - ``"backbone"`` / ``"backbone_freeze"``: ``{"N", "CA", "C", "O"}``
    - ``"ca"`` / ``"c_alpha"``: ``{"CA"}``
    - ``"heavy"`` / ``"af2_heavy_atom"``: all non-hydrogen heavy atoms
    - ``"none"``: no atoms restrained

    Solvent and ion residues (``{"HOH", "WAT", "TIP3", "NA", "CL", "K", "MG", "ZN"}``
    plus any ``exclude_residue_names``) are automatically excluded.
    """
    sel = selection.strip().lower()
    if sel in ("backbone", "backbone_freeze"):
        mode = "backbone"
    elif sel in ("ca", "c_alpha"):
        mode = "ca"
    elif sel in ("heavy", "af2_heavy_atom"):
        mode = "heavy"
    elif sel == "none":
        mode = "none"
    else:
        raise ValueError(
            f"Unsupported restraint selection {selection!r}; expected one of "
            "('backbone', 'ca', 'heavy', 'none')."
        )

    excluded_res = set(SOLVENT_AND_ION_RESIDUES)
    if exclude_residue_names is not None:
        excluded_res.update(r.strip().upper() for r in exclude_residue_names)

    n_atoms = len(atom_names)
    if residue_names is not None and len(residue_names) != n_atoms:
        raise ValueError(
            f"Length mismatch: len(atom_names)={n_atoms} vs len(residue_names)={len(residue_names)}"
        )

    flags: list[bool] = []
    for idx, raw_name in enumerate(atom_names):
        if mode == "none":
            flags.append(False)
            continue
        if residue_names is not None:
            res_name = residue_names[idx].strip().upper()
            if res_name in excluded_res:
                flags.append(False)
                continue
        atom_name = raw_name.strip().upper()
        if mode == "backbone":
            flags.append(atom_name in BACKBONE_ATOM_NAMES)
        elif mode == "ca":
            flags.append(atom_name in CA_ATOM_NAMES)
        else:
            flags.append(_is_heavy_atom_name(atom_name))

    out = jnp.asarray(flags, dtype=jnp.bool_)
    if atom_mask is not None:
        out = out & jnp.asarray(atom_mask, dtype=jnp.bool_)
    return out


def _invoke_energy_stage_fn(
    energy_stage_fn: Callable[..., Array],
    positions: Array,
    *,
    lambda_lj: float,
    lambda_elec: float,
    pme_enabled: bool,
    dispersion_tail_enabled: bool,
    softcore_alpha: float,
    softcore_exponent: int,
    soft_coulomb: bool,
) -> Array:
    """Invoke ``energy_stage_fn`` with all supported stage control kwargs."""
    sig = inspect.signature(energy_stage_fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    candidate_kwargs = {
        "lambda_lj": jnp.asarray(lambda_lj, dtype=positions.dtype),
        "lambda_elec": jnp.asarray(lambda_elec, dtype=positions.dtype),
        "use_pme_reciprocal": pme_enabled,
        "use_pme_exclusion_correction": pme_enabled,
        "use_dispersion_correction": dispersion_tail_enabled,
        "softcore_alpha": softcore_alpha,
        "softcore_exponent": softcore_exponent,
        "soft_coulomb": soft_coulomb,
    }
    if accepts_var_kw:
        return energy_stage_fn(positions, **candidate_kwargs)

    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in params}
    if filtered_kwargs:
        return energy_stage_fn(positions, **filtered_kwargs)

    # Positional fallback: (positions, lambda_lj[, lambda_elec])
    pos_params = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(pos_params) >= 3:
        return energy_stage_fn(
            positions,
            jnp.asarray(lambda_lj, dtype=positions.dtype),
            jnp.asarray(lambda_elec, dtype=positions.dtype),
        )
    if len(pos_params) == 2:
        return energy_stage_fn(positions, jnp.asarray(lambda_lj, dtype=positions.dtype))
    return energy_stage_fn(positions)


def run_staged_softcore_minimization(
    energy_stage_fn: Callable[..., Array],
    initial_positions: Array,
    reference_positions: Array,
    restraint_mask: Array | None,
    displacement_fn: Callable[[Array, Array], Array],
    *,
    windows: Sequence[SoftcoreWindow] = DEFAULT_NAURMALADE_WINDOWS,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    soft_coulomb: bool = False,
    k_restraint_kcal_mol_a2: float = 2.390057,
    flat_bottom_radius_angstrom: float = 0.0,
    step_size: float = 1e-3,
    max_grad_norm: float = 10.0,
) -> tuple[Array, list[dict[str, float]]]:
    """Run staged Beutler soft-core minimization across a sequence of lambda windows (`FR-8`).

    During soft-core windows (`lambda_lj < 1.0`), PME reciprocal-space, PME
    exclusion corrections, and the isotropic LJ dispersion tail are disabled
    (matching `naurmalade._build_softcore_working_system`), and electrostatics
    default to ``lambda_elec = 0.0`` unless ``soft_coulomb=True``. At the final
    full-Hamiltonian stage (`lambda_lj == 1.0`), full electrostatics and PME/tail
    settings are enabled.

    Args:
        energy_stage_fn: Callable evaluating system potential energy at a given stage.
        initial_positions: Starting coordinates ``(N, 3)`` in Angstroms.
        reference_positions: Restraint reference coordinates ``(N, 3)`` in Angstroms.
        restraint_mask: Boolean/float ``(N,)`` mask for restrained atoms (or ``None``).
        displacement_fn: PBC or free-space displacement function ``(r_i, r_j) -> dr``.
        windows: Sequence of :class:`SoftcoreWindow` stages.
        softcore_alpha: Beutler soft-core ``alpha_sc`` (default ``0.5``).
        softcore_exponent: Beutler exponent ``p`` on ``(1 - lambda_lj)^p`` (default ``2``).
        soft_coulomb: Whether to evaluate soft-core cutoff Coulomb during soft-core windows.
        k_restraint_kcal_mol_a2: Positional restraint spring constant in ``kcal/(mol*A^2)``
            (default ``1000.0 / 418.4 = 2.390057``).
        flat_bottom_radius_angstrom: Flat-bottom restraint radius in Angstroms.
        step_size: Initial gradient-descent step size in Angstroms.
        max_grad_norm: Per-atom force norm clipping ceiling in ``kcal/(mol*A)``.

    Returns:
        Tuple ``(final_positions, stage_history)`` where ``stage_history`` contains
        per-window telemetry dictionaries.
    """
    pos = jnp.asarray(initial_positions)
    ref_pos = jnp.asarray(reference_positions, dtype=pos.dtype)
    if restraint_mask is None:
        rest_mask = jnp.zeros((pos.shape[0],), dtype=pos.dtype)
    else:
        rest_mask = jnp.asarray(restraint_mask, dtype=pos.dtype)

    history: list[dict[str, float]] = []

    for window in windows:
        is_full_hamiltonian = float(window.lambda_lj) >= 1.0 - 1e-7
        effective_lambda_elec = (
            1.0
            if (is_full_hamiltonian and window.lambda_elec == 0.0)
            else (float(window.lambda_elec) if (soft_coulomb or is_full_hamiltonian) else 0.0)
        )
        pme_enabled = is_full_hamiltonian
        dispersion_tail_enabled = is_full_hamiltonian

        def _stage_total_energy(r_curr: Array) -> Array:
            e_phys = _invoke_energy_stage_fn(
                energy_stage_fn,
                r_curr,
                lambda_lj=float(window.lambda_lj),
                lambda_elec=effective_lambda_elec,
                pme_enabled=pme_enabled,
                dispersion_tail_enabled=dispersion_tail_enabled,
                softcore_alpha=softcore_alpha,
                softcore_exponent=softcore_exponent,
                soft_coulomb=soft_coulomb,
            )
            e_rest = compute_positional_restraint_energy(
                r_curr,
                ref_pos,
                rest_mask,
                displacement_fn,
                k_kcal_mol_a2=k_restraint_kcal_mol_a2,
                flat_bottom_radius_angstrom=flat_bottom_radius_angstrom,
                restraint_scale=window.restraint_scale,
            )
            return e_phys + e_rest

        val_and_grad_fn = jax.value_and_grad(_stage_total_energy)
        init_e, grad = val_and_grad_fn(pos)
        curr_e = init_e
        lr = float(step_size)

        for _ in range(int(window.max_iterations)):
            grad_clean = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))
            atom_norms = jnp.sqrt(jnp.sum(grad_clean * grad_clean, axis=-1, keepdims=True) + 1e-12)
            scale = jnp.minimum(1.0, float(max_grad_norm) / atom_norms)
            clipped_grad = grad_clean * scale

            candidate_pos = pos - lr * clipped_grad
            cand_e, cand_grad = val_and_grad_fn(candidate_pos)

            if bool(jnp.isfinite(cand_e)) and float(cand_e) <= float(curr_e) + 1e-5:
                pos = candidate_pos
                curr_e = cand_e
                grad = cand_grad
                lr = min(lr * 1.1, float(step_size) * 5.0)
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break

        grad_clean = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))
        force_norms = jnp.sqrt(jnp.sum(grad_clean * grad_clean, axis=-1))
        rms_grad = float(jnp.sqrt(jnp.mean(force_norms * force_norms)))
        peak_force = float(jnp.max(force_norms))

        history.append(
            {
                "lambda_lj": float(window.lambda_lj),
                "lambda_elec": float(effective_lambda_elec),
                "restraint_scale": float(window.restraint_scale),
                "initial_energy": float(init_e),
                "final_energy": float(curr_e),
                "rms_grad": rms_grad,
                "peak_force": peak_force,
                "pme_enabled": float(1.0 if pme_enabled else 0.0),
                "dispersion_tail_enabled": float(1.0 if dispersion_tail_enabled else 0.0),
            }
        )

    return pos, history


@dataclass(frozen=True)
class RestraintSpec:
    """Declarative positional, distance, and dihedral restraint specification."""

    selection: str = "backbone"
    k_kj_nm2: float = 1000.0
    flat_bottom_radius_angstrom: float = 0.0
    distance_pairs: Array | None = None
    distance_targets_angstrom: Array | None = None
    distance_k_kcal_mol_a2: float = 0.0
    dihedral_quads: Array | None = None
    dihedral_targets_rad: Array | None = None
    dihedral_k_kcal_mol_rad2: float = 0.0

    @property
    def k_kcal_mol_a2(self) -> float:
        return k_kj_nm2_to_kcal_a2(self.k_kj_nm2)


def compute_distance_restraint_energy(
    positions: Array,
    pairs: Array,
    target_distances: Array,
    displacement_fn: Callable[[Array, Array], Array],
    *,
    k_kcal_mol_a2: float | Array = 0.0,
    flat_bottom_radius_angstrom: float = 0.0,
    mask: Array | None = None,
) -> Array:
    """Compute flat-bottom harmonic distance restraint energy over atom pairs."""
    pos = jnp.asarray(positions)
    idx = jnp.asarray(pairs, dtype=jnp.int32)
    if idx.shape[0] == 0:
        return jnp.zeros((), dtype=pos.dtype)
    dtype = pos.dtype
    r0 = jnp.asarray(target_distances, dtype=dtype)
    m = jnp.ones((idx.shape[0],), dtype=dtype) if mask is None else jnp.asarray(mask, dtype=dtype)
    dr = jax.vmap(displacement_fn)(pos[idx[:, 0]], pos[idx[:, 1]])
    dist = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + jnp.asarray(1e-12, dtype=dtype))
    dev = jnp.abs(dist - r0)
    excess = jnp.maximum(jnp.asarray(0.0, dtype=dtype), dev - jnp.asarray(flat_bottom_radius_angstrom, dtype=dtype))
    return jnp.asarray(k_kcal_mol_a2, dtype=dtype) * jnp.sum(m * excess * excess)


def compute_dihedral_restraint_energy(
    positions: Array,
    quads: Array,
    target_angles_rad: Array,
    displacement_fn: Callable[[Array, Array], Array],
    *,
    k_kcal_mol_rad2: float | Array = 0.0,
    mask: Array | None = None,
) -> Array:
    """Compute periodic-wrapped harmonic dihedral restraint energy over 4-atom quadruplets."""
    pos = jnp.asarray(positions)
    idx = jnp.asarray(quads, dtype=jnp.int32)
    if idx.shape[0] == 0:
        return jnp.zeros((), dtype=pos.dtype)
    dtype = pos.dtype
    phi = bonded.compute_dihedral_angles(pos, idx, displacement_fn)
    phi0 = jnp.asarray(target_angles_rad, dtype=dtype)
    m = jnp.ones((idx.shape[0],), dtype=dtype) if mask is None else jnp.asarray(mask, dtype=dtype)
    diff = jnp.arctan2(jnp.sin(phi - phi0), jnp.cos(phi - phi0))
    return jnp.asarray(k_kcal_mol_rad2, dtype=dtype) * jnp.sum(m * diff * diff)
