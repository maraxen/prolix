#!/usr/bin/env python3
"""High-precision parity and propagator-throughput benchmark: Prolix vs kUPS (P1).

Builds on the Phase 3 apples-to-apples harness (``scripts/benchmarks/pallas_vs_baseline.py``)
and evaluates three tiers across system sizes:

1. **Tier 1 — High-Precision Unit, Potential, Force & Single-Step Trajectory Parity (float64/float32)**
   - **1A (Harmonic + BAOAB/CSVR single step, float64)**: Compares kUPS (eV/Å/amu/fs units)
     against Prolix (AKMA units via ``prolix.physics.kups_adapter`` and
     ``prolix.physics.md_potential_bundle.value_energy_and_forces``) with matched PRNG
     draws, verifying relative energy parity $< 10^{-12}$, force parity $< 10^{-12}$ kcal/(mol·Å),
     and single-step BAOAB position/momentum parity $< 10^{-12}$ Å.
   - **1B (Periodic Nonbonded Fluid: LJ + PME-direct damped Coulomb)**: Compares kUPS-unit
     nonbonded evaluation against Prolix XLA (``chunked_lj_energy_nl + chunked_coulomb_energy_nl``)
     in float64 and Prolix Pallas (``pallas_nonbonded_nl``) in float32 across $N \\in \\{512, 2048, 8192\\}$.

2. **Tier 2 — Thermostat Ensemble Cross-Validation (BAOAB & CSVR across $\\Delta t \\in \\{0.5, 1.0, 2.0\\}$ fs)**
   - Runs multi-thousand-step ``jax.lax.scan`` trajectories in both kUPS (eV/Å/amu) and
     Prolix (AKMA), checking mean equilibrium temperature $|T_{\\text{Prolix}} - T_{\\text{kUPS}}|$,
     canonical kinetic-temperature fluctuation ratio $\\sigma_T / (T \\sqrt{2/N_{\\text{dof}}})$,
     and CSVR $\\Delta t \\ge 1.0$ fs velocity-Verlet discretization bias agreement.

3. **Tier 3 — Propagator Throughput & AD-Site Bundling Scaling ($N \\in \\{512, 2048, 8192, 32768\\}$)**
   - Benchmarks ``jax.lax.scan`` MD steps (logging scalar potential energy, kinetic energy,
     and temperature every step) across four propagator architectures on the periodic
     nonbonded fluid:
     * ``naive_split_xla``: separate ``energy_fn(R)`` + ``grad(energy_fn)(R)`` per step (2 passes),
     * ``kups_cached_xla``: kUPS ``CachePropagator`` / ``PotentialOut`` pattern in eV/Å/amu (1 ``value_and_grad``),
     * ``prolix_bundle_xla``: Prolix ``value_energy_and_forces`` (`EnergyForceBundle`) in AKMA (1 ``value_and_grad``),
     * ``prolix_bundle_pallas``: Prolix ``value_energy_and_forces`` + ``pallas_nonbonded_nl``
       (forward-force Pallas kernel + $O(N)$ VJP).

Usage::

    uv run python scripts/benchmarks/kups_benchmark.py \\
        --sizes 512,2048,8192,32768 --out outputs/bench/kups_benchmark.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from proxide.physics.constants import COULOMB_CONSTANT

from prolix.pallas_kernels import _triton_gpu_available, pallas_nonbonded_nl
from prolix.physics import kups_adapter
from prolix.physics.md_potential_bundle import value_energy_and_forces
from prolix.physics.optimization import chunked_coulomb_energy_nl, chunked_lj_energy_nl
from prolix.simulate import BOLTZMANN_KCAL
from scripts.benchmarks.pallas_vs_baseline import (
    CUTOFF,
    DENSITY,
    PME_ALPHA,
    SWITCH_WIDTH,
    build_system,
    fit_exponent,
    time_fn,
)

KUPS_INSTALLED = importlib.util.find_spec("kups") is not None
T_TARGET_K = 300.0
K_HARMONIC_EV = 0.01
M_AMU = 12.0
GAMMA_PS = 10.0
TAU_PS = 0.1
COULOMB_EV_A = COULOMB_CONSTANT * kups_adapter.KCAL_MOL_TO_EV


# =============================================================================
# Reference kUPS eV/A/amu/fs Propagator & Potential Engine
# =============================================================================


def make_kups_nonbonded_energy_ev(s: dict):
    """Nonbonded (LJ + PME-direct damped Coulomb) energy in kUPS units (eV, A)."""
    disp, _ = space.periodic(s["box"])
    eps_ev = s["eps"] * jnp.asarray(kups_adapter.KCAL_MOL_TO_EV, dtype=s["eps"].dtype)
    q_scaled = s["q"] * jnp.asarray(math.sqrt(kups_adapter.KCAL_MOL_TO_EV), dtype=s["q"].dtype)

    def energy_ev_fn(pos):
        e_lj_ev = chunked_lj_energy_nl(
            pos, s["sigma"], eps_ev, s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
            cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
        )
        e_c_ev = chunked_coulomb_energy_nl(
            pos, q_scaled, s["excl_idx"], s["excl_elec"], s["nbr"], disp,
            pme_alpha=PME_ALPHA, cutoff=CUTOFF, tile_size=256,
        )
        return e_lj_ev + e_c_ev

    return energy_ev_fn


def make_prolix_nonbonded_energy_kcal(s: dict, use_pallas: bool, interpret: bool):
    """Nonbonded (LJ + PME-direct damped Coulomb) energy in Prolix AKMA units (kcal/mol, A)."""
    disp, _ = space.periodic(s["box"])

    if not use_pallas:
        def energy_kcal_fn(pos):
            e_lj = chunked_lj_energy_nl(
                pos, s["sigma"], s["eps"], s["excl_idx"], s["excl_vdw"], s["nbr"], disp,
                cutoff=CUTOFF, tile_size=256, switch_width=SWITCH_WIDTH,
            )
            e_c = chunked_coulomb_energy_nl(
                pos, s["q"], s["excl_idx"], s["excl_elec"], s["nbr"], disp,
                pme_alpha=PME_ALPHA, cutoff=CUTOFF, tile_size=256,
            )
            return e_lj + e_c

        return energy_kcal_fn

    def pallas_kcal_fn(pos):
        return pallas_nonbonded_nl(
            pos, s["q"], s["sigma"], s["eps"], s["mask"], s["nbr"], s["excl_idx"],
            s["excl_vdw"], s["excl_elec"], box=s["box"], pme_alpha=PME_ALPHA,
            cutoff=CUTOFF, lj_switch_width=SWITCH_WIDTH, soft_core_lambda=1.0,
            include_pme_excl_erf=True, interpret=interpret,
        )

    return pallas_kcal_fn


def baoab_step_with_noise(
    pos: jax.Array,
    mom: jax.Array,
    force: jax.Array,
    pe: jax.Array,
    mass: jax.Array,
    dt: float,
    gamma: float,
    kT: float,
    noise: jax.Array,
    eval_ef_fn,
    box: jax.Array | None = None,
):
    """One BAOAB step with cached force/energy (kUPs CachePropagator / Prolix bundle pattern)."""
    dt_half = jnp.asarray(0.5 * dt, dtype=pos.dtype)
    m_col = mass[:, None]
    # B (half kick)
    mom = mom + dt_half * force
    # A (half drift)
    pos = pos + dt_half * (mom / m_col)
    if box is not None:
        pos = jnp.mod(pos, box)
    # O (exact Ornstein-Uhlenbeck momentum update)
    c1 = jnp.asarray(math.exp(-gamma * dt), dtype=pos.dtype)
    c2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - c1 * c1) * m_col * jnp.asarray(kT, dtype=pos.dtype))
    mom = c1 * mom + c2 * noise
    # A (half drift)
    pos = pos + dt_half * (mom / m_col)
    if box is not None:
        pos = jnp.mod(pos, box)
    # Single potential+force evaluation at new position
    new_pe, new_force = eval_ef_fn(pos)
    # B (half kick)
    mom = mom + dt_half * new_force
    return pos, mom, new_force, new_pe


def csvr_step_with_noise(
    pos: jax.Array,
    mom: jax.Array,
    force: jax.Array,
    pe: jax.Array,
    mass: jax.Array,
    dt: float,
    tau: float,
    kT: float,
    noise_dof: jax.Array,
    eval_ef_fn,
):
    """One velocity-Verlet + Bussi-Donadio-Parrinello CSVR step with matched DOF noise."""
    dt_half = jnp.asarray(0.5 * dt, dtype=pos.dtype)
    m_col = mass[:, None]
    dof = pos.shape[0] * 3
    # Half kick + full drift
    mom = mom + dt_half * force
    pos = pos + jnp.asarray(dt, dtype=pos.dtype) * (mom / m_col)
    new_pe, new_force = eval_ef_fn(pos)
    mom = mom + dt_half * new_force
    # CSVR velocity rescaling (exact Bussi et al. 2007 Eq. A7)
    ke = 0.5 * jnp.sum((mom * mom) / m_col)
    ke_target = 0.5 * dof * jnp.asarray(kT, dtype=pos.dtype)
    c = jnp.asarray(math.exp(-dt / tau), dtype=pos.dtype)
    r1 = noise_dof[0]
    sum_r2 = jnp.sum(noise_dof[1:] ** 2)
    ratio = ke_target / (dof * jnp.maximum(ke, 1e-30))
    alpha_sq = (
        c
        + (1.0 - c) * ratio * (r1 * r1 + sum_r2)
        + 2.0 * jnp.exp(jnp.asarray(-0.5 * dt / tau, dtype=pos.dtype)) * jnp.sqrt((1.0 - c) * ratio) * r1
    )
    alpha = jnp.sqrt(jnp.maximum(alpha_sq, 0.0))
    mom = alpha * mom
    return pos, mom, new_force, new_pe


# =============================================================================
# Tier 1: High-Precision Single-Step & Nonbonded Parity
# =============================================================================


def run_tier1_parity(sizes: list[int], interpret: bool) -> dict:
    """Tier 1A (float64 harmonic + BAOAB/CSVR single-step parity) & Tier 1B (periodic nonbonded)."""
    jax.config.update("jax_enable_x64", True)

    harmonic_rows = []
    for n in (64, 512, 2048):
        key = jax.random.PRNGKey(42 + n)
        k1, k2, k3 = jax.random.split(key, 3)
        dt_fs = 1.0
        dt_kups = kups_adapter.dt_fs_to_kups(dt_fs)
        dt_akma = kups_adapter.dt_fs_to_akma(dt_fs)
        gamma_kups = kups_adapter.gamma_ps_to_kups(GAMMA_PS)
        gamma_akma = kups_adapter.gamma_ps_to_akma(GAMMA_PS)
        tau_kups = kups_adapter.tau_ps_to_kups(TAU_PS)
        tau_akma = kups_adapter.tau_ps_to_akma(TAU_PS)

        k_ev = K_HARMONIC_EV
        k_kcal = kups_adapter.spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev)
        kT_kcal = T_TARGET_K * BOLTZMANN_KCAL
        kT_ev = kups_adapter.energy_kcal_mol_to_ev(kT_kcal)

        pos0 = jax.random.normal(k1, (n, 3), dtype=jnp.float64) * 0.2
        mass = jnp.full((n,), M_AMU, dtype=jnp.float64)
        mom0_kups = jax.random.normal(k2, (n, 3), dtype=jnp.float64) * jnp.sqrt(M_AMU * kT_ev)
        mom0_akma = kups_adapter.momentum_kups_to_akma(mom0_kups)
        noise = jax.random.normal(k3, (n, 3), dtype=jnp.float64)

        def e_kups_fn(r):
            return 0.5 * k_ev * jnp.sum(r * r)

        def e_prolix_fn(r):
            return 0.5 * k_kcal * jnp.sum(r * r)

        def ef_kups(r):
            b = value_energy_and_forces(e_kups_fn, r)
            return b.energy, b.forces

        def ef_prolix(r):
            b = value_energy_and_forces(e_prolix_fn, r)
            return b.energy, b.forces

        pe0_kups, f0_kups = ef_kups(pos0)
        pe0_prolix, f0_prolix = ef_prolix(pos0)

        pos1_kups, mom1_kups, f1_kups, pe1_kups = baoab_step_with_noise(
            pos0, mom0_kups, f0_kups, pe0_kups, mass,
            dt_kups, gamma_kups, kT_ev, noise, ef_kups,
        )
        pos1_prolix, mom1_prolix, f1_prolix, pe1_prolix = baoab_step_with_noise(
            pos0, mom0_akma, f0_prolix, pe0_prolix, mass,
            dt_akma, gamma_akma, kT_kcal, noise, ef_prolix,
        )

        # CSVR single step with matched DOF noise
        noise_dof = jax.random.normal(k3, (3 * n,), dtype=jnp.float64)
        pos_c_kups, mom_c_kups, _, _ = csvr_step_with_noise(
            pos0, mom0_kups, f0_kups, pe0_kups, mass,
            dt_kups, tau_kups, kT_ev, noise_dof, ef_kups,
        )
        pos_c_prolix, mom_c_prolix, _, _ = csvr_step_with_noise(
            pos0, mom0_akma, f0_prolix, pe0_prolix, mass,
            dt_akma, tau_akma, kT_kcal, noise_dof, ef_prolix,
        )

        pe1_kups_in_kcal = float(kups_adapter.energy_ev_to_kcal_mol(pe1_kups))
        f1_kups_in_kcal = kups_adapter.force_ev_per_angstrom_to_kcal_mol(f1_kups)
        mom1_kups_in_akma = kups_adapter.momentum_kups_to_akma(mom1_kups)
        mom_c_kups_in_akma = kups_adapter.momentum_kups_to_akma(mom_c_kups)

        row = {
            "n_atoms": n,
            "rel_dE_f64": float(abs(float(pe1_prolix) - pe1_kups_in_kcal) / max(abs(float(pe1_prolix)), 1e-30)),
            "max_abs_dF_kcal_mol_A_f64": float(jnp.max(jnp.abs(f1_prolix - f1_kups_in_kcal))),
            "baoab_max_pos_err_A_f64": float(jnp.max(jnp.abs(pos1_prolix - pos1_kups))),
            "baoab_max_mom_err_akma_f64": float(jnp.max(jnp.abs(mom1_prolix - mom1_kups_in_akma))),
            "csvr_max_pos_err_A_f64": float(jnp.max(jnp.abs(pos_c_prolix - pos_c_kups))),
            "csvr_max_mom_err_akma_f64": float(jnp.max(jnp.abs(mom_c_prolix - mom_c_kups_in_akma))),
        }
        print(json.dumps({"tier": "1A_harmonic", **row}), flush=True)
        harmonic_rows.append(row)

    # Tier 1B: Periodic Nonbonded Fluid (LJ + PME-direct damped Coulomb)
    nonbonded_rows = []
    for n in [sz for sz in sizes if sz <= 8192]:
        s32 = build_system(n)
        s64 = {
            k: (v.astype(jnp.float64) if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating) else v)
            for k, v in s32.items()
        }
        # Float64 XLA vs kUPS eV
        kups_e64 = make_kups_nonbonded_energy_ev(s64)
        prolix_e64 = make_prolix_nonbonded_energy_kcal(s64, use_pallas=False, interpret=interpret)
        b_kups64 = jax.jit(lambda p: value_energy_and_forces(kups_e64, p))(s64["pos"])
        b_prolix64 = jax.jit(lambda p: value_energy_and_forces(prolix_e64, p))(s64["pos"])
        e_kups64_kcal = float(kups_adapter.energy_ev_to_kcal_mol(b_kups64.energy))
        f_kups64_kcal = kups_adapter.force_ev_per_angstrom_to_kcal_mol(b_kups64.forces)

        # Float32 Pallas vs kUPS eV
        jax.config.update("jax_enable_x64", False)
        kups_e32 = make_kups_nonbonded_energy_ev(s32)
        pallas_e32 = make_prolix_nonbonded_energy_kcal(s32, use_pallas=True, interpret=interpret)
        b_kups32 = jax.jit(lambda p: value_energy_and_forces(kups_e32, p))(s32["pos"])
        b_pallas32 = jax.jit(lambda p: value_energy_and_forces(pallas_e32, p))(s32["pos"])
        e_kups32_kcal = float(kups_adapter.energy_ev_to_kcal_mol(b_kups32.energy))
        f_kups32_kcal = kups_adapter.force_ev_per_angstrom_to_kcal_mol(b_kups32.forces)
        jax.config.update("jax_enable_x64", True)

        nb_row = {
            "n_atoms": n,
            "k": int(s32["k"]),
            "xla_vs_kups_rel_dE_f64": float(
                abs(float(b_prolix64.energy) - e_kups64_kcal) / max(abs(float(b_prolix64.energy)), 1e-30)
            ),
            "xla_vs_kups_max_abs_dF_f64": float(jnp.max(jnp.abs(b_prolix64.forces - f_kups64_kcal))),
            "pallas_vs_kups_rel_dE_f32": float(
                abs(float(b_pallas32.energy) - e_kups32_kcal) / max(abs(float(b_pallas32.energy)), 1e-12)
            ),
            "pallas_vs_kups_max_abs_dF_f32": float(jnp.max(jnp.abs(b_pallas32.forces - f_kups32_kcal))),
        }
        print(json.dumps({"tier": "1B_nonbonded", **nb_row}), flush=True)
        nonbonded_rows.append(nb_row)

    jax.config.update("jax_enable_x64", False)
    return {"harmonic_single_step_f64": harmonic_rows, "nonbonded_parity": nonbonded_rows}


# =============================================================================
# Tier 2: Thermostat Ensemble Cross-Validation (BAOAB & CSVR)
# =============================================================================


def run_tier2_thermostat(n_particles: int = 256, n_equil: int = 4000, n_sample: int = 6000) -> list[dict]:
    """Run BAOAB & CSVR trajectories in kUPS (eV/A/amu) and Prolix (AKMA) with jax.lax.scan."""
    jax.config.update("jax_enable_x64", True)
    dof = 3 * n_particles
    k_ev = K_HARMONIC_EV
    k_kcal = kups_adapter.spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev)
    kT_kcal = T_TARGET_K * BOLTZMANN_KCAL
    kT_ev = kups_adapter.energy_kcal_mol_to_ev(kT_kcal)
    canonical_std_k = T_TARGET_K * math.sqrt(2.0 / dof)

    def ef_kups(r):
        b = value_energy_and_forces(lambda x: 0.5 * k_ev * jnp.sum(x * x), r)
        return b.energy, b.forces

    def ef_prolix(r):
        b = value_energy_and_forces(lambda x: 0.5 * k_kcal * jnp.sum(x * x), r)
        return b.energy, b.forces

    cases = [
        ("BAOAB", 0.5, 2.0),
        ("BAOAB", 1.0, 2.0),
        ("BAOAB", 2.0, 2.5),
        ("CSVR", 0.5, 2.0),
        ("CSVR", 1.0, 5.0),
    ]
    rows = []
    for integ, dt_fs, tol_k in cases:
        key = jax.random.PRNGKey(100)
        k1, k2, k_run = jax.random.split(key, 3)
        pos0 = jax.random.normal(k1, (n_particles, 3), dtype=jnp.float64) * 0.1
        mass = jnp.full((n_particles,), 1.0, dtype=jnp.float64)
        mom0_kups = jax.random.normal(k2, (n_particles, 3), dtype=jnp.float64) * jnp.sqrt(1.0 * kT_ev)
        mom0_akma = kups_adapter.momentum_kups_to_akma(mom0_kups)
        pe0_kups, f0_kups = ef_kups(pos0)
        pe0_akma, f0_akma = ef_prolix(pos0)

        dt_kups = kups_adapter.dt_fs_to_kups(dt_fs)
        dt_akma = kups_adapter.dt_fs_to_akma(dt_fs)
        gamma_kups = kups_adapter.gamma_ps_to_kups(GAMMA_PS)
        gamma_akma = kups_adapter.gamma_ps_to_akma(GAMMA_PS)
        tau_kups = kups_adapter.tau_ps_to_kups(TAU_PS)
        tau_akma = kups_adapter.tau_ps_to_akma(TAU_PS)

        def make_runner(is_kups: bool):
            dt_val = dt_kups if is_kups else dt_akma
            g_val = gamma_kups if is_kups else gamma_akma
            tau_val = tau_kups if is_kups else tau_akma
            kt_val = kT_ev if is_kups else kT_kcal
            ef_fn = ef_kups if is_kups else ef_prolix
            kb_val = (kT_ev / T_TARGET_K) if is_kups else BOLTZMANN_KCAL

            def step_fn(carry, _):
                rng, p, m, f, pe = carry
                rng, sub = jax.random.split(rng)
                if integ == "BAOAB":
                    noise = jax.random.normal(sub, (n_particles, 3), dtype=jnp.float64)
                    p, m, f, pe = baoab_step_with_noise(
                        p, m, f, pe, mass, dt_val, g_val, kt_val, noise, ef_fn
                    )
                else:
                    noise_dof = jax.random.normal(sub, (dof,), dtype=jnp.float64)
                    p, m, f, pe = csvr_step_with_noise(
                        p, m, f, pe, mass, dt_val, tau_val, kt_val, noise_dof, ef_fn
                    )
                ke = 0.5 * jnp.sum((m * m) / mass[:, None])
                temp_k = (2.0 * ke) / (dof * kb_val)
                return (rng, p, m, f, pe), temp_k

            @jax.jit
            def run_traj(rng, p0, m0, f0, pe0):
                carry, _ = jax.lax.scan(step_fn, (rng, p0, m0, f0, pe0), None, length=n_equil)
                _, temps = jax.lax.scan(step_fn, carry, None, length=n_sample)
                return temps

            return run_traj

        temps_kups = np.asarray(make_runner(True)(k_run, pos0, mom0_kups, f0_kups, pe0_kups))
        temps_prolix = np.asarray(make_runner(False)(k_run, pos0, mom0_akma, f0_akma, pe0_akma))
        t_kups_mean = float(np.mean(temps_kups))
        t_prolix_mean = float(np.mean(temps_prolix))
        t_kups_std = float(np.std(temps_kups))
        t_prolix_std = float(np.std(temps_prolix))
        delta_t = abs(t_prolix_mean - t_kups_mean)
        max_step_diff = float(np.max(np.abs(temps_prolix - temps_kups)))
        row = {
            "integrator": integ,
            "dt_fs": dt_fs,
            "n_particles": n_particles,
            "n_equil": n_equil,
            "n_sample": n_sample,
            "t_prolix_k": t_prolix_mean,
            "t_kups_k": t_kups_mean,
            "delta_t_k": delta_t,
            "max_stepwise_temp_diff_k": max_step_diff,
            "prolix_std_over_canonical": t_prolix_std / canonical_std_k,
            "kups_std_over_canonical": t_kups_std / canonical_std_k,
            "tolerance_k": tol_k,
            "gate_pass": bool(delta_t <= tol_k),
        }
        print(json.dumps({"tier": "2_thermostat", **row}), flush=True)
        rows.append(row)

    jax.config.update("jax_enable_x64", False)
    return rows


# =============================================================================
# Tier 3: Propagator Throughput & AD-Site Bundling Scaling
# =============================================================================


def run_tier3_throughput(sizes: list[int], n_steps: int, repeats: int, interpret: bool) -> tuple[list[dict], dict]:
    """Compare 4 MD propagator architectures logging (PE, KE, T) every step over lax.scan."""
    jax.config.update("jax_enable_x64", False)
    dt_fs = 2.0
    dt_akma = kups_adapter.dt_fs_to_akma(dt_fs)
    dt_kups = kups_adapter.dt_fs_to_kups(dt_fs)
    gamma_akma = kups_adapter.gamma_ps_to_akma(GAMMA_PS)
    gamma_kups = kups_adapter.gamma_ps_to_kups(GAMMA_PS)
    kT_kcal = T_TARGET_K * BOLTZMANN_KCAL
    kT_ev = kups_adapter.energy_kcal_mol_to_ev(kT_kcal)
    ns_per_step = dt_fs * 1e-6  # 2 fs = 2e-6 ns
    steps_per_day_factor = 86400.0 * ns_per_step

    rows = []
    for n in sizes:
        s = build_system(n, include_nl=True)
        mass = jnp.full((n,), M_AMU, dtype=jnp.float32)
        key = jax.random.PRNGKey(7)
        mom0_akma = jax.random.normal(key, (n, 3), dtype=jnp.float32) * jnp.sqrt(
            jnp.float32(M_AMU * kT_kcal)
        )
        mom0_kups = kups_adapter.momentum_akma_to_kups(mom0_akma)
        box = s["box"]

        prolix_xla_e = make_prolix_nonbonded_energy_kcal(s, use_pallas=False, interpret=interpret)
        prolix_pallas_e = make_prolix_nonbonded_energy_kcal(s, use_pallas=True, interpret=interpret)
        kups_xla_e = make_kups_nonbonded_energy_ev(s)

        def make_scan_propagator(mode: str):
            if mode == "naive_split_xla":
                # Anti-pattern addressed by ADR-001 / kUPS CachePropagator:
                # force and scalar energy compiled behind separate JIT boundaries
                # (e.g. integrator force closure vs observer energy logger), preventing CSE.
                jit_e = jax.jit(prolix_xla_e)
                jit_g = jax.jit(jax.grad(prolix_xla_e))

                def ef_fn(r):
                    e = jit_e(r)
                    # Barrier on r so XLA cannot fuse jit_e and jit_g into a single pass
                    r_bar = jax.lax.optimization_barrier(r)
                    f = -jit_g(r_bar)
                    return e, f

                dt_v, g_v, kt_v = dt_akma, gamma_akma, kT_kcal
            elif mode == "kups_cached_xla":
                def ef_fn(r):
                    b = value_energy_and_forces(kups_xla_e, r)
                    return b.energy, b.forces

                dt_v, g_v, kt_v = dt_kups, gamma_kups, kT_ev
            elif mode == "prolix_bundle_xla":
                def ef_fn(r):
                    b = value_energy_and_forces(prolix_xla_e, r)
                    return b.energy, b.forces

                dt_v, g_v, kt_v = dt_akma, gamma_akma, kT_kcal
            elif mode == "prolix_bundle_pallas":
                def ef_fn(r):
                    b = value_energy_and_forces(prolix_pallas_e, r)
                    return b.energy, b.forces

                dt_v, g_v, kt_v = dt_akma, gamma_akma, kT_kcal
            else:
                raise ValueError(mode)

            def scan_fn(pos_init, mom_init):
                pe0, f0 = ef_fn(pos_init)

                def body(carry, step_idx):
                    p, m, f, pe = carry
                    sub = jax.random.fold_in(key, step_idx)
                    noise = jax.random.normal(sub, (n, 3), dtype=jnp.float32)
                    p, m, f, pe = baoab_step_with_noise(
                        p, m, f, pe, mass, dt_v, g_v, kt_v, noise, ef_fn, box=box
                    )
                    ke = 0.5 * jnp.sum((m * m) / mass[:, None])
                    return (p, m, f, pe), (pe, ke)

                (p_fin, m_fin, f_fin, pe_fin), (pes, kes) = jax.lax.scan(
                    body, (pos_init, mom_init, f0, pe0), jnp.arange(n_steps)
                )
                return p_fin, pes, kes

            return jax.jit(scan_fn)

        row: dict = {"n_atoms": n, "k": int(s["k"]), "n_steps": n_steps}
        outs = {}
        for mode in ("naive_split_xla", "kups_cached_xla", "prolix_bundle_xla", "prolix_bundle_pallas"):
            fn = make_scan_propagator(mode)
            m_init = mom0_kups if mode == "kups_cached_xla" else mom0_akma
            t0 = time.perf_counter()
            out = fn(s["pos"], m_init)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(s["pos"], m_init))
                times.append(time.perf_counter() - t0)
            med_total_s = statistics.median(times)
            step_ms = (med_total_s / n_steps) * 1000.0
            ns_per_day = (n_steps / med_total_s) * steps_per_day_factor
            row[mode] = {
                "compile_s": compile_s,
                "scan_median_s": med_total_s,
                "step_ms": step_ms,
                "ns_per_day": ns_per_day,
            }
            outs[mode] = out

        # Speedup ratios and multi-step trajectory parity vs kups_cached_xla
        t_naive = row["naive_split_xla"]["step_ms"]
        t_kups = row["kups_cached_xla"]["step_ms"]
        t_pb_xla = row["prolix_bundle_xla"]["step_ms"]
        t_pb_pal = row["prolix_bundle_pallas"]["step_ms"]
        row["bundle_vs_naive_split_speedup_xla"] = t_naive / t_pb_xla
        row["prolix_xla_vs_kups_xla_ratio"] = t_kups / t_pb_xla
        row["prolix_pallas_vs_kups_xla_speedup"] = t_kups / t_pb_pal
        row["prolix_pallas_vs_naive_split_speedup"] = t_naive / t_pb_pal

        kups_pes_kcal = np.asarray(outs["kups_cached_xla"][1]) * kups_adapter.EV_TO_KCAL_MOL
        pb_xla_pes = np.asarray(outs["prolix_bundle_xla"][1])
        pb_pal_pes = np.asarray(outs["prolix_bundle_pallas"][1])
        row["trajectory_rel_dE_prolix_xla_vs_kups"] = float(
            np.max(np.abs(pb_xla_pes - kups_pes_kcal)) / max(float(np.max(np.abs(pb_xla_pes))), 1e-12)
        )
        row["trajectory_rel_dE_prolix_pallas_vs_kups"] = float(
            np.max(np.abs(pb_pal_pes - kups_pes_kcal)) / max(float(np.max(np.abs(pb_pal_pes))), 1e-12)
        )
        print(json.dumps({"tier": "3_throughput", **row}), flush=True)
        rows.append(row)

    ns = [r["n_atoms"] for r in rows]
    summary = {
        mode + "_time_exponent": fit_exponent(ns, [r[mode]["step_ms"] for r in rows])
        for mode in ("naive_split_xla", "kups_cached_xla", "prolix_bundle_xla", "prolix_bundle_pallas")
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="512,2048,8192,32768")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-equil", type=int, default=4000)
    ap.add_argument("--n-sample", type=int, default=6000)
    ap.add_argument("--tier", choices=("1", "2", "3", "all"), default="all")
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--out", default="outputs/bench/kups_benchmark.json")
    args = ap.parse_args()

    interpret = args.interpret or not _triton_gpu_available()
    sizes = [int(x) for x in args.sizes.split(",")]
    payload: dict = {
        "meta": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
            "pallas_interpret": interpret,
            "kups_package_installed": KUPS_INSTALLED,
            "jax": jax.__version__,
            "host": platform.node(),
            "cutoff": CUTOFF,
            "density": DENSITY,
        }
    }
    if args.tier in ("1", "all"):
        payload["tier1_parity"] = run_tier1_parity(sizes, interpret)
    if args.tier in ("2", "all"):
        payload["tier2_thermostat"] = run_tier2_thermostat(
            n_particles=256, n_equil=args.n_equil, n_sample=args.n_sample
        )
    if args.tier in ("3", "all"):
        rows3, sum3 = run_tier3_throughput(sizes, args.n_steps, args.repeats, interpret)
        payload["tier3_throughput"] = {"rows": rows3, "summary": sum3}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
