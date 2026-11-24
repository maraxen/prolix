# ruff: noqa: E402
"""Turnkey Pallas v3.6 / v3.7 A/B Physics & Throughput Benchmark Suite.

Evaluates five A/B tracks side-by-side against OpenMM baselines:
  - S1: NPT TIP3P reciprocal-pressure virial (+353 bar fixed-G bias vs exact
        isotropic SPME Green's scaling) and NPT equilibrium liquid density.
  - S2: NVT HBonds solute-water equipartition delta_T and symplectic RATTLE
        velocity projection (pre-B2 vs post-B2 radial velocity & rotational KE).
  - S3: Suite D OpenMM frontend fast-path (ProlixContext.step GPU Morton runner
        vs host-sync fallback) and ProlixSimulation staged reporter speedup.
  - S4: DHFR_24k / nb>=64 Morton 3D cell-grid + 8-octant sub-AABB pruned tile
        pair counts and step latency.
  - S5: BSD-3-Clause Pallas fltflt (hi, lo) rn.f32 mixed-precision vs single
        precision step ratio and holonomic bond constraint conservation.

On GPU runs (default), OpenMM baselines strictly enforce `require_openmm_cuda_context`
(`libOpenMMCUDA.so` mapped, zero silent OpenCL/CPU/Reference fallback). Pass
`--cpu-smoke` for deterministic CI smoke verification without GPU hardware.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
  sys.path.insert(0, str(ROOT / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import openmm as mm
from openmm import app
from openmm import unit as _omm_unit

from prolix.api.ensemble_plan import EnsemblePlan
from prolix.api.observables import KCAL_MOL_PER_A3_TO_BAR
from prolix.constants import BOLTZMANN_KCAL_PER_MOL_K
from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation, from_openmm_system
from prolix.physics.pallas_tile_pairs import (
  _build_tile_pairs_jax_fallback,
  _compute_tile_aabbs,
  _morton3d_encode_jax,
  estimate_max_tile_pairs,
  pallas_build_morton_tile_pairs_gpu,
  precompute_dedup_excl_edges,
)
from prolix.physics.pme import (
  influence_function,
  make_spme_energy_fn,
  spme_background_energy,
  spme_energy_with_forces,
  spme_self_energy,
  spread_charges,
)
from prolix.physics.settle import (
  TIP3P_RHH,
  TIP3P_ROH,
  pallas_baoab_post_force,
  pallas_baoab_pre_force,
)
from scripts.benchmarks.openmm_cuda_guard import (
  extract_openmm_context_provenance,
  get_openmm_platform_inventory,
  require_openmm_cuda_context,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

unit = cast("Any", _omm_unit)

AVOGADRO_MOL_INV = 6.02214076e23
AMU_PER_A3_TO_G_CM3 = 1.0e24 / AVOGADRO_MOL_INV  # ~1.66053906660
AKMA_TIME_FS = 48.88821291839


def _openmm_context_for_mode(
  system: mm.System,
  integrator: mm.Integrator,
  *,
  expected_precision: str = "single",
  cpu_smoke: bool = False,
) -> tuple[mm.Context, dict[str, Any]]:
  """Create a guarded OpenMM CUDA context on GPU or Reference/CPU context in `--cpu-smoke`."""
  if not cpu_smoke:
    return require_openmm_cuda_context(
      system,
      integrator,
      expected_precision=expected_precision,
      allow_opencl_fallback=False,
      allow_cpu_fallback=False,
    )
  inv = get_openmm_platform_inventory()
  plat_name = "CPU" if "CPU" in inv["platforms"] else "Reference"
  plat = mm.Platform.getPlatformByName(plat_name)
  ctx = mm.Context(system, integrator, plat)
  return ctx, extract_openmm_context_provenance(ctx)


def _build_tip3p_openmm_box(
  n_waters: int,
  *,
  with_barostat: bool = False,
  cutoff_nm: float = 0.85,
  pme_alpha_nm_inv: float = 3.4,
  pme_grid: int = 24,
  barostat_interval: int = 5,
  box_nm_override: float | None = None,
  seed: int = 20261117,
) -> tuple[mm.System, app.Topology, np.ndarray, float]:
  """Construct an orthorhombic rigid-TIP3P OpenMM System, Topology, and coordinates (nm)."""
  # Liquid water target density ~0.997 g/cm^3 -> ~30.05 A^3 per water molecule
  if box_nm_override is not None:
    box_nm = float(box_nm_override)
    box_a = box_nm * 10.0
  else:
    min_mic_box_a = 2.0 * cutoff_nm * 10.0 + 1.5
    vol_a3 = max(float(n_waters) * 30.05, min_mic_box_a**3)
    box_a = float(vol_a3 ** (1.0 / 3.0))
    box_nm = box_a * 0.1

  top = app.Topology()
  chain = top.addChain()
  elem_o = app.Element.getBySymbol("O")
  elem_h = app.Element.getBySymbol("H")

  sys_mm = mm.System()
  sys_mm.setDefaultPeriodicBoxVectors(
    mm.Vec3(box_nm, 0.0, 0.0),
    mm.Vec3(0.0, box_nm, 0.0),
    mm.Vec3(0.0, 0.0, box_nm),
  )
  nb = mm.NonbondedForce()
  nb.setNonbondedMethod(mm.NonbondedForce.PME)
  nb.setCutoffDistance(cutoff_nm * unit.nanometer)
  nb.setPMEParameters(pme_alpha_nm_inv, pme_grid, pme_grid, pme_grid)
  nb.setUseDispersionCorrection(True)

  n_side = max(1, math.ceil(n_waters ** (1.0 / 3.0)))
  spacing_nm = box_nm / float(n_side)
  rng = np.random.default_rng(seed)
  theta = math.radians(104.52)
  r_oh_nm = 0.09572
  r_hh_nm = 0.15139

  coords_nm: list[np.ndarray] = []
  for w in range(n_waters):
    res = top.addResidue("HOH", chain)
    a_o = top.addAtom("O", elem_o, res)
    a_h1 = top.addAtom("H1", elem_h, res)
    a_h2 = top.addAtom("H2", elem_h, res)
    top.addBond(a_o, a_h1)
    top.addBond(a_o, a_h2)

    ix = w % n_side
    iy = (w // n_side) % n_side
    iz = w // (n_side * n_side)
    jitter = rng.uniform(-0.015, 0.015, size=(3,))
    o_pos = np.array(
      [(ix + 0.5) * spacing_nm, (iy + 0.5) * spacing_nm, (iz + 0.5) * spacing_nm],
      dtype=np.float64,
    ) + jitter
    o_pos = np.mod(o_pos, box_nm)
    h1_pos = o_pos + np.array([r_oh_nm, 0.0, 0.0], dtype=np.float64)
    h2_pos = o_pos + np.array(
      [r_oh_nm * math.cos(theta), r_oh_nm * math.sin(theta), 0.0],
      dtype=np.float64,
    )
    coords_nm.extend([o_pos, h1_pos, h2_pos])

    base = 3 * w
    sys_mm.addParticle(15.9994)
    sys_mm.addParticle(1.00794)
    sys_mm.addParticle(1.00794)
    sys_mm.addConstraint(base, base + 1, r_oh_nm * unit.nanometer)
    sys_mm.addConstraint(base, base + 2, r_oh_nm * unit.nanometer)
    sys_mm.addConstraint(base + 1, base + 2, r_hh_nm * unit.nanometer)

    nb.addParticle(-0.834, 0.315075, 0.635968)
    nb.addParticle(0.417, 0.04, 0.0)
    nb.addParticle(0.417, 0.04, 0.0)
    nb.addException(base, base + 1, 0.0, 0.1, 0.0)
    nb.addException(base, base + 2, 0.0, 0.1, 0.0)
    nb.addException(base + 1, base + 2, 0.0, 0.1, 0.0)

  sys_mm.addForce(nb)
  if with_barostat:
    sys_mm.addForce(
      mm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, barostat_interval)
    )
  top.setPeriodicBoxVectors(sys_mm.getDefaultPeriodicBoxVectors())
  return sys_mm, top, np.asarray(coords_nm, dtype=np.float64), box_a


def _legacy_fixed_g_spme_energy(
  positions: jax.Array,
  charges: jax.Array,
  atom_mask: jax.Array,
  box_cur: jax.Array,
  g_ref_fixed: jax.Array,
  grid_dims: tuple[int, int, int],
  alpha: float,
  order: int = 4,
) -> jax.Array:
  """Reproduce pre-v3.6 SPME cached energy where G(box_ref) was scaled only by V_ref/V_cur."""
  q_mesh = spread_charges(positions, charges, atom_mask, box_cur, grid_dims, order, False)
  q_hat = jnp.fft.rfftn(q_mesh, s=grid_dims, axes=(-3, -2, -1))
  theta = jnp.fft.irfftn(q_hat * g_ref_fixed, s=grid_dims, axes=(-3, -2, -1))
  n_grid = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
  e_rec = 0.5 * 332.06371329919216 * jnp.sum(q_mesh * theta) * n_grid
  e_self = spme_self_energy(charges, atom_mask, alpha)
  e_bg = spme_background_energy(charges, atom_mask, alpha, box_cur)
  return e_rec + e_self + e_bg


# ============================================================================
# S1: NPT TIP3P Reciprocal Pressure (+353 bar Fix) & Liquid Density
# ============================================================================
def run_suite_s1_npt_tip3p(
  *,
  cpu_smoke: bool = False,
  quick: bool = False,
) -> dict[str, Any]:
  """Benchmark S1: A/B reciprocal virial pressure and NPT TIP3P equilibrium density."""
  # Part A: Exact molecular-COM volume-scaling reciprocal pressure check (64 rigid waters)
  rng = np.random.default_rng(20261117)
  n_w_vir = 64
  l0 = 18.6
  box_ref = jnp.array([l0, l0, l0], dtype=jnp.float64)
  grid_dims = (24, 24, 24)
  alpha = 0.34

  coms = rng.uniform(0.0, l0, size=(n_w_vir, 3))
  theta_hoh = math.radians(104.52)
  h1_off = np.array([TIP3P_ROH, 0.0, 0.0], dtype=np.float64)
  h2_off = np.array(
    [TIP3P_ROH * math.cos(theta_hoh), TIP3P_ROH * math.sin(theta_hoh), 0.0],
    dtype=np.float64,
  )

  def _scaled_water_positions(scale: float) -> jax.Array:
    sc = coms * scale
    pos = np.empty((n_w_vir * 3, 3), dtype=np.float64)
    pos[0::3] = sc
    pos[1::3] = sc + h1_off
    pos[2::3] = sc + h2_off
    return jnp.asarray(pos, dtype=jnp.float64)

  charges = jnp.asarray(np.tile([-0.834, 0.417, 0.417], n_w_vir), dtype=jnp.float64)
  mask = jnp.ones((n_w_vir * 3,), dtype=jnp.float64)
  g_ref_unscaled = influence_function(grid_dims, box_ref, alpha, order=4)
  cached_v36_fn = make_spme_energy_fn(
    box_ref, alpha=alpha, order=4, use_pallas_pme=False, grid_dims=grid_dims
  )

  delta = 1.0e-4
  v_plus = (l0 * (1.0 + delta)) ** 3
  v_minus = (l0 * (1.0 - delta)) ** 3
  dv = v_plus - v_minus
  pos_plus = _scaled_water_positions(1.0 + delta)
  pos_minus = _scaled_water_positions(1.0 - delta)
  box_plus = box_ref * (1.0 + delta)
  box_minus = box_ref * (1.0 - delta)

  # Pre-v3.6 bug: cached G(box_ref) reused unscaled across trial volume moves
  e_leg_plus = float(
    _legacy_fixed_g_spme_energy(pos_plus, charges, mask, box_plus, g_ref_unscaled, grid_dims, alpha)
  )
  e_leg_minus = float(
    _legacy_fixed_g_spme_energy(pos_minus, charges, mask, box_minus, g_ref_unscaled, grid_dims, alpha)
  )
  e_v36_plus = float(cached_v36_fn(pos_plus, charges, mask, box=box_plus))
  e_v36_minus = float(cached_v36_fn(pos_minus, charges, mask, box=box_minus))
  e_exact_plus = float(
    spme_energy_with_forces(pos_plus, charges, mask, box_plus, grid_dims, alpha, 4, False)
  )
  e_exact_minus = float(
    spme_energy_with_forces(pos_minus, charges, mask, box_minus, grid_dims, alpha, 4, False)
  )

  p_legacy_bar = -((e_leg_plus - e_leg_minus) / dv) * KCAL_MOL_PER_A3_TO_BAR
  p_v36_bar = -((e_v36_plus - e_v36_minus) / dv) * KCAL_MOL_PER_A3_TO_BAR
  p_exact_bar = -((e_exact_plus - e_exact_minus) / dv) * KCAL_MOL_PER_A3_TO_BAR
  legacy_bias_bar = p_legacy_bar - p_exact_bar
  v36_abs_error_bar = abs(p_v36_bar - p_exact_bar)

  # Part B: NPT TIP3P liquid density rollout (Prolix vs OpenMM)
  n_waters_npt = 125 if cpu_smoke else (128 if quick else 256)
  n_steps_npt = 4 if cpu_smoke else (200 if quick else 1000)
  baro_interval = 2 if cpu_smoke else 10
  sys_npt, _, pos_nm_npt, _ = _build_tip3p_openmm_box(
    n_waters_npt,
    with_barostat=True,
    cutoff_nm=0.65 if cpu_smoke else 0.85,
    pme_grid=16 if cpu_smoke else 24,
    barostat_interval=baro_interval,
  )

  integ_omm = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 2.0 / unit.picosecond, 0.001 * unit.picoseconds
  )
  ctx_omm, prov_omm = _openmm_context_for_mode(
    sys_npt, integ_omm, expected_precision="single", cpu_smoke=cpu_smoke
  )
  ctx_omm.setPositions(pos_nm_npt * unit.nanometer)
  ctx_omm.setVelocitiesToTemperature(300.0 * unit.kelvin, 101)
  integ_omm.step(n_steps_npt)
  st_omm = ctx_omm.getState(getPositions=True)
  vol_omm_a3 = float(st_omm.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)) * 1000.0
  total_mass_amu = float(n_waters_npt) * (15.9994 + 2.0 * 1.00794)
  omm_density = (total_mass_amu / vol_omm_a3) * AMU_PER_A3_TO_G_CM3
  del ctx_omm, integ_omm

  cfg_prx = ProlixOpenMMConfig(
    precision="single",
    use_pallas=not cpu_smoke,
    pallas_interpret=cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nonbonded_engine="morton_tile_pairs" if not cpu_smoke else "neighbor_list",
    nl_update_every=4 if cpu_smoke else 10,
    barostat_mode="mc",
  )
  integ_prx = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 2.0 / unit.picosecond, 0.001 * unit.picoseconds
  )
  ctx_prx = ProlixContext(sys_npt, integ_prx, config=cfg_prx)
  ctx_prx.setPositions(pos_nm_npt * unit.nanometer)
  ctx_prx.setVelocitiesToTemperature(300.0 * unit.kelvin, 101)
  ctx_prx.step(n_steps_npt)
  st_prx = ctx_prx.getState(getPositions=True)
  vol_prx_a3 = float(st_prx.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)) * 1000.0
  prx_density = (total_mass_amu / vol_prx_a3) * AMU_PER_A3_TO_G_CM3

  return {
    "n_waters_virial": n_w_vir,
    "legacy_unscaled_recip_pressure_bar": float(p_legacy_bar),
    "v36_scaled_recip_pressure_bar": float(p_v36_bar),
    "exact_spme_recip_pressure_bar": float(p_exact_bar),
    "legacy_bias_bar": float(legacy_bias_bar),
    "v36_abs_error_bar": float(v36_abs_error_bar),
    "n_waters_npt": n_waters_npt,
    "n_steps_npt": n_steps_npt,
    "prolix_mean_density_g_cm3": float(prx_density),
    "openmm_mean_density_g_cm3": float(omm_density),
    "density_diff_g_cm3": float(prx_density - omm_density),
    "density_rel_pct": float(100.0 * (prx_density - omm_density) / max(omm_density, 1e-12)),
    "openmm_provenance": prov_omm,
  }


# ============================================================================
# S2: NVT HBonds Solute-Water Equipartition delta_T & Symplectic RATTLE
# ============================================================================
def _build_solvated_hbonds_solute_water_system(
  n_waters: int = 16,
) -> tuple[mm.System, np.ndarray, int, int]:
  """Build a constrained CH3-CH2-like solute + rigid TIP3P waters in a periodic box."""
  box_nm = max(2.2, 0.1 * ((float(n_waters + 2) * 32.0) ** (1.0 / 3.0)))
  sys_mm = mm.System()
  sys_mm.setDefaultPeriodicBoxVectors(
    mm.Vec3(box_nm, 0.0, 0.0),
    mm.Vec3(0.0, box_nm, 0.0),
    mm.Vec3(0.0, 0.0, box_nm),
  )
  nb = mm.NonbondedForce()
  nb.setNonbondedMethod(mm.NonbondedForce.PME)
  nb.setCutoffDistance(0.75 * unit.nanometer)
  nb.setPMEParameters(3.4, 20, 20, 20)
  bonds = mm.HarmonicBondForce()
  angles = mm.HarmonicAngleForce()

  # 6-atom constrained ethane-like solute: C0-C1 harmonic bond, 4 C-H holonomic constraints
  d_cc = 0.1526
  d_ch = 0.1090
  c0 = np.array([0.45, 0.45, 0.45], dtype=np.float64)
  c1 = c0 + np.array([d_cc, 0.0, 0.0], dtype=np.float64)
  solute_coords = [
    c0,
    c0 + np.array([0.0, d_ch, 0.0]),
    c0 + np.array([0.0, -0.5 * d_ch, 0.8660254 * d_ch]),
    c1,
    c1 + np.array([0.0, -d_ch, 0.0]),
    c1 + np.array([0.0, 0.5 * d_ch, -0.8660254 * d_ch]),
  ]
  solute_masses = [12.011, 1.008, 1.008, 12.011, 1.008, 1.008]
  solute_q = [-0.12, 0.06, 0.06, -0.12, 0.06, 0.06]
  solute_sig = [0.34, 0.26, 0.26, 0.34, 0.26, 0.26]
  solute_eps = [0.45, 0.06, 0.06, 0.45, 0.06, 0.06]
  for m, q, s, e in zip(solute_masses, solute_q, solute_sig, solute_eps, strict=True):
    sys_mm.addParticle(m)
    nb.addParticle(q, s, e)

  bonds.addBond(0, 3, d_cc * unit.nanometer, 250000.0)
  angles.addAngle(1, 0, 3, 1.91, 400.0)
  angles.addAngle(2, 0, 3, 1.91, 400.0)
  angles.addAngle(0, 3, 4, 1.91, 400.0)
  angles.addAngle(0, 3, 5, 1.91, 400.0)
  # 4 X-H constraints on solute
  for heavy, hydro in ((0, 1), (0, 2), (3, 4), (3, 5)):
    sys_mm.addConstraint(heavy, hydro, d_ch * unit.nanometer)
  solute_bonds = [(0, 3), (0, 1), (0, 2), (3, 4), (3, 5)]
  nb.createExceptionsFromBonds(solute_bonds, 0.8333333333, 0.5)

  n_solute = 6
  n_solute_constraints = 4
  coords = list(solute_coords)
  theta = math.radians(104.52)
  n_side = max(2, math.ceil((n_waters + 2) ** (1.0 / 3.0)))
  spacing = box_nm / float(n_side)
  placed = 0
  grid_idx = 0
  while placed < n_waters:
    ix = grid_idx % n_side
    iy = (grid_idx // n_side) % n_side
    iz = grid_idx // (n_side * n_side)
    grid_idx += 1
    o_pos = np.array(
      [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing],
      dtype=np.float64,
    )
    if np.linalg.norm(o_pos - c0) < 0.45:
      continue
    h1_pos = o_pos + np.array([0.09572, 0.0, 0.0])
    h2_pos = o_pos + np.array([0.09572 * math.cos(theta), 0.09572 * math.sin(theta), 0.0])
    coords.extend([o_pos, h1_pos, h2_pos])

    base = n_solute + 3 * placed
    sys_mm.addParticle(15.9994)
    sys_mm.addParticle(1.00794)
    sys_mm.addParticle(1.00794)
    sys_mm.addConstraint(base, base + 1, 0.09572 * unit.nanometer)
    sys_mm.addConstraint(base, base + 2, 0.09572 * unit.nanometer)
    sys_mm.addConstraint(base + 1, base + 2, 0.15139 * unit.nanometer)
    nb.addParticle(-0.834, 0.315075, 0.635968)
    nb.addParticle(0.417, 0.04, 0.0)
    nb.addParticle(0.417, 0.04, 0.0)
    nb.addException(base, base + 1, 0.0, 0.1, 0.0)
    nb.addException(base, base + 2, 0.0, 0.1, 0.0)
    nb.addException(base + 1, base + 2, 0.0, 0.1, 0.0)
    placed += 1

  sys_mm.addForce(bonds)
  sys_mm.addForce(angles)
  sys_mm.addForce(nb)
  return sys_mm, np.asarray(coords, dtype=np.float64), n_solute, n_solute_constraints


def run_suite_s2_nvt_hbonds_delta_t(
  *,
  cpu_smoke: bool = False,
  quick: bool = False,
) -> dict[str, Any]:
  """Benchmark S2: RATTLE rotational KE conservation + NVT solute-water delta_T."""
  # Part A: Single-step RATTLE rotational KE & post-B2 radial orthogonality check
  d_nh = 1.01
  tet_z = -d_nh / 3.0
  tet_r = d_nh * math.sqrt(8.0 / 9.0)
  r_n = np.array([5.0, 5.0, 5.0], dtype=np.float64)
  r_h1 = r_n + np.array([tet_r, 0.0, tet_z])
  r_h2 = r_n + np.array([-0.5 * tet_r, 0.5 * math.sqrt(3.0) * tet_r, tet_z])
  r_h3 = r_n + np.array([-0.5 * tet_r, -0.5 * math.sqrt(3.0) * tet_r, tet_z])
  half_hh = 0.5 * TIP3P_RHH
  y_h = math.sqrt(TIP3P_ROH * TIP3P_ROH - half_hh * half_hh)
  r_o = np.array([9.5, 5.0, 5.0], dtype=np.float64)
  off_h1 = np.array([half_hh, y_h, 0.0], dtype=np.float64)
  off_h2 = np.array([-half_hh, y_h, 0.0], dtype=np.float64)
  pos0_np = np.stack(
    [r_n, r_h1, r_h2, r_h3, r_o, r_o + off_h1, r_o + off_h2],
    axis=0,
  )
  mass_np = np.array([14.007, 1.008, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=np.float64)
  omega = np.array([0.0, 0.0, 0.35], dtype=np.float64)
  vel0_np = np.zeros_like(pos0_np)
  for h_idx in (1, 2, 3):
    vel0_np[h_idx] = np.cross(omega, pos0_np[h_idx] - pos0_np[0])

  pos0 = jnp.asarray(pos0_np, dtype=jnp.float64)
  mass_j = jnp.asarray(mass_np, dtype=jnp.float64)
  mom0 = jnp.asarray(vel0_np * mass_np[:, None], dtype=jnp.float64)
  wi = jnp.asarray([[4, 5, 6]], dtype=jnp.int32)
  wm = jnp.asarray([1], dtype=jnp.int32)
  cp = jnp.asarray([[0, 1], [0, 2], [0, 3]], dtype=jnp.int32)
  cl = jnp.asarray([d_nh, d_nh, d_nh], dtype=jnp.float64)
  cm = jnp.asarray([1, 1, 1], dtype=jnp.int32)
  f_zero = jnp.zeros_like(pos0)
  dt_akma = 2.0 / AKMA_TIME_FS

  pos1, mom_pre, _ = pallas_baoab_pre_force(
    pos0,
    mom0,
    f_zero,
    mass_j,
    jax.random.key(0),
    wi,
    wm,
    box=None,
    dt=dt_akma,
    kT=0.0,
    gamma=0.0,
    interpret=True,
    constraint_pairs=cp,
    constraint_lengths=cl,
    constraint_mask=cm,
    f64_accumulators=True,
  )
  mom1 = pallas_baoab_post_force(
    pos1,
    mom_pre,
    f_zero,
    mass_j,
    wi,
    wm,
    dt=dt_akma,
    interpret=True,
    constraint_pairs=cp,
    constraint_lengths=cl,
    constraint_mask=cm,
    f64_accumulators=True,
  )
  ke0 = float(0.5 * jnp.sum((mom0 * mom0) / mass_j[:, None]))
  ke1 = float(0.5 * jnp.sum((mom1 * mom1) / mass_j[:, None]))
  rot_ke_rel_err = abs(ke1 - ke0) / max(ke0, 1e-12)

  pos1_np = np.asarray(pos1, dtype=np.float64)
  vel1_np = np.asarray(mom1, dtype=np.float64) / mass_np[:, None]
  dr_c = pos1_np[np.asarray(cp)[:, 1]] - pos1_np[np.asarray(cp)[:, 0]]
  u_c = dr_c / np.linalg.norm(dr_c, axis=-1, keepdims=True)
  dv_c = vel1_np[np.asarray(cp)[:, 1]] - vel1_np[np.asarray(cp)[:, 0]]
  post_b2_rad_vel_err = float(np.max(np.abs(np.sum(dv_c * u_c, axis=-1))))

  # Part B: Solvated HBonds NVT solute-water kinetic temperature split
  n_wat = 16 if cpu_smoke else (96 if quick else 192)
  n_samples = 4 if cpu_smoke else (20 if quick else 50)
  steps_per_sample = 3 if cpu_smoke else 25
  sys_nv, pos_nv, n_sol, n_sol_cons = _build_solvated_hbonds_solute_water_system(n_waters=n_wat)

  masses_amu = np.array(
    [sys_nv.getParticleMass(i).value_in_unit(unit.dalton) for i in range(sys_nv.getNumParticles())],
    dtype=np.float64,
  )
  dof_solute = 3 * n_sol - n_sol_cons
  dof_water = 6 * n_wat  # 9 - 3 SETTLE constraints per water

  def _temperatures_from_vel_nm_ps(vel_nm_ps: np.ndarray) -> tuple[float, float]:
    # KE in kJ/mol = 0.5 * m (Da) * v^2 (nm/ps)^2; k_B = 0.008314462618 kJ/(mol*K)
    ke_per_atom_kj = 0.5 * masses_amu * np.sum(vel_nm_ps * vel_nm_ps, axis=-1)
    kb_kj = 0.00831446261815324
    t_sol = float(2.0 * np.sum(ke_per_atom_kj[:n_sol]) / (dof_solute * kb_kj))
    t_wat = float(2.0 * np.sum(ke_per_atom_kj[n_sol:]) / (dof_water * kb_kj))
    return t_sol, t_wat

  integ_o = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 5.0 / unit.picosecond, 0.001 * unit.picoseconds
  )
  ctx_o, prov_o = _openmm_context_for_mode(
    sys_nv, integ_o, expected_precision="mixed", cpu_smoke=cpu_smoke
  )
  ctx_o.setPositions(pos_nv * unit.nanometer)
  ctx_o.setVelocitiesToTemperature(300.0 * unit.kelvin, 202)
  omm_t_sol_samples: list[float] = []
  omm_t_wat_samples: list[float] = []
  for _ in range(n_samples):
    integ_o.step(steps_per_sample)
    v_o = (
      ctx_o.getState(getVelocities=True)
      .getVelocities(asNumpy=True)
      .value_in_unit(unit.nanometer / unit.picosecond)
    )
    ts, tw = _temperatures_from_vel_nm_ps(v_o)
    omm_t_sol_samples.append(ts)
    omm_t_wat_samples.append(tw)
  del ctx_o, integ_o

  cfg_p = ProlixOpenMMConfig(
    precision="mixed",
    use_pallas=not cpu_smoke,
    pallas_interpret=cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nl_update_every=5 if cpu_smoke else 15,
  )
  integ_p = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 5.0 / unit.picosecond, 0.001 * unit.picoseconds
  )
  ctx_p = ProlixContext(sys_nv, integ_p, config=cfg_p)
  ctx_p.setPositions(pos_nv * unit.nanometer)
  ctx_p.setVelocitiesToTemperature(300.0 * unit.kelvin, 202)
  prx_t_sol_samples: list[float] = []
  prx_t_wat_samples: list[float] = []
  for _ in range(n_samples):
    ctx_p.step(steps_per_sample)
    v_p = (
      ctx_p.getState(getVelocities=True)
      .getVelocities(asNumpy=True)
      .value_in_unit(unit.nanometer / unit.picosecond)
    )
    ts, tw = _temperatures_from_vel_nm_ps(v_p)
    prx_t_sol_samples.append(ts)
    prx_t_wat_samples.append(tw)

  t_sol_mean = float(np.mean(prx_t_sol_samples))
  t_wat_mean = float(np.mean(prx_t_wat_samples))
  omm_sol_mean = float(np.mean(omm_t_sol_samples))
  omm_wat_mean = float(np.mean(omm_t_wat_samples))
  return {
    "rotational_ke_rel_err": float(rot_ke_rel_err),
    "post_b2_rad_vel_err": float(post_b2_rad_vel_err),
    "n_waters": n_wat,
    "n_samples": n_samples,
    "steps_per_sample": steps_per_sample,
    "t_solute_k": t_sol_mean,
    "t_water_k": t_wat_mean,
    "delta_t_solute_minus_water_k": float(t_sol_mean - t_wat_mean),
    "openmm_t_solute_k": omm_sol_mean,
    "openmm_t_water_k": omm_wat_mean,
    "openmm_delta_t_k": float(omm_sol_mean - omm_wat_mean),
    "openmm_provenance": prov_o,
  }


# ============================================================================
# S3: Suite D ProlixContext.step Fast-Path & ProlixSimulation Staged Reporters
# ============================================================================
def run_suite_s3_openmm_fastpath_and_reporters(
  *,
  cpu_smoke: bool = False,
  quick: bool = False,
) -> dict[str, Any]:
  """Benchmark S3: GPU Morton ProlixContext.step fast-path + staged reporter speedup."""
  n_waters = 16 if cpu_smoke else (256 if quick else 512)
  n_warmup = 2 if cpu_smoke else 20
  n_steps = 8 if cpu_smoke else (100 if quick else 400)
  report_interval = 2 if cpu_smoke else 10
  n_reports = max(2, n_steps // report_interval)

  sys_s3, top_s3, pos_s3, _ = _build_tip3p_openmm_box(
    n_waters,
    with_barostat=False,
    cutoff_nm=0.75 if cpu_smoke else 0.85,
    pme_grid=16 if cpu_smoke else 32,
  )

  # OpenMM baseline
  integ_omm = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
  )
  ctx_omm, prov_omm = _openmm_context_for_mode(
    sys_s3, integ_omm, expected_precision="single", cpu_smoke=cpu_smoke
  )
  ctx_omm.setPositions(pos_s3 * unit.nanometer)
  ctx_omm.setVelocitiesToTemperature(300.0 * unit.kelvin, 303)
  integ_omm.step(n_warmup)
  _ = ctx_omm.getState(getEnergy=True)
  t_omm0 = time.perf_counter()
  integ_omm.step(n_steps)
  _ = ctx_omm.getState(getEnergy=True)
  omm_ms_step = ((time.perf_counter() - t_omm0) / n_steps) * 1000.0
  del ctx_omm, integ_omm

  # Arm A: Legacy pure-JAX host-NL fallback (use_pallas=False)
  cfg_legacy = ProlixOpenMMConfig(
    precision="single",
    use_pallas=False,
    use_pallas_pme=False,
    nonbonded_engine="neighbor_list",
    nl_update_every=4 if cpu_smoke else 10,
  )
  ctx_leg = ProlixContext(
    sys_s3,
    mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    ),
    config=cfg_legacy,
  )
  ctx_leg.setPositions(pos_s3 * unit.nanometer)
  ctx_leg.setVelocitiesToTemperature(300.0 * unit.kelvin, 303)
  ctx_leg.step(n_warmup)
  _ = ctx_leg.getState(getEnergy=True)
  t_leg0 = time.perf_counter()
  ctx_leg.step(n_steps)
  _ = ctx_leg.getState(getEnergy=True)
  legacy_ms_step = ((time.perf_counter() - t_leg0) / n_steps) * 1000.0

  # Arm B: v3.6 GPU-resident Morton fast-path
  cfg_fast = ProlixOpenMMConfig(
    precision="single",
    use_pallas=True,
    pallas_interpret=cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nonbonded_engine="morton_tile_pairs",
    nl_update_every=4 if cpu_smoke else 20,
  )
  ctx_fast = ProlixContext(
    sys_s3,
    mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    ),
    config=cfg_fast,
  )
  ctx_fast.setPositions(pos_s3 * unit.nanometer)
  ctx_fast.setVelocitiesToTemperature(300.0 * unit.kelvin, 303)
  ctx_fast.step(n_warmup)
  _ = ctx_fast.getState(getEnergy=True, getPositions=True)
  morton_active = getattr(ctx_fast, "_morton_runner", None) is not None
  t_fast0 = time.perf_counter()
  ctx_fast.step(n_steps)
  _ = ctx_fast.getState(getEnergy=True, getPositions=True)
  fast_ms_step = ((time.perf_counter() - t_fast0) / n_steps) * 1000.0

  # Part 2: ProlixSimulation reporter A/B (per-interval host sync vs staged Simulation.step)
  sim_unbatched = ProlixSimulation(
    top_s3,
    sys_s3,
    mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    ),
    config=cfg_fast,
  )
  sim_unbatched.context.setPositions(pos_s3 * unit.nanometer)
  sim_unbatched.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 304)
  sim_unbatched.step(n_warmup)
  t_rep_a0 = time.perf_counter()
  for _ in range(n_reports):
    sim_unbatched.context.step(report_interval)
    _ = sim_unbatched.context.getState(getPositions=True, getEnergy=True)
  unbatched_rep_ms = ((time.perf_counter() - t_rep_a0) / (n_reports * report_interval)) * 1000.0

  buf_csv = io.StringIO()
  sim_staged = ProlixSimulation(
    top_s3,
    sys_s3,
    mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    ),
    config=cfg_fast,
  )
  sim_staged.context.setPositions(pos_s3 * unit.nanometer)
  sim_staged.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 304)
  sim_staged.step(n_warmup)
  sim_staged.reporters.append(
    app.StateDataReporter(
      buf_csv,
      reportInterval=report_interval,
      step=True,
      potentialEnergy=True,
      kineticEnergy=True,
    )
  )
  t_rep_b0 = time.perf_counter()
  sim_staged.step(n_reports * report_interval)
  staged_rep_ms = ((time.perf_counter() - t_rep_b0) / (n_reports * report_interval)) * 1000.0
  data_lines = [
    ln for ln in buf_csv.getvalue().splitlines() if ln.strip() and not ln.startswith("#")
  ]

  return {
    "n_waters": n_waters,
    "n_steps": n_steps,
    "morton_fastpath_active": bool(morton_active),
    "legacy_fallback_ms_per_step": float(legacy_ms_step),
    "v36_fastpath_ms_per_step": float(fast_ms_step),
    "fastpath_speedup_x": float(legacy_ms_step / max(fast_ms_step, 1e-12)),
    "openmm_ms_per_step": float(omm_ms_step),
    "report_interval": report_interval,
    "n_reports_emitted": len(data_lines),
    "unbatched_reporter_ms_per_step": float(unbatched_rep_ms),
    "staged_simulation_ms_per_step": float(staged_rep_ms),
    "reporter_speedup_x": float(unbatched_rep_ms / max(staged_rep_ms, 1e-12)),
    "openmm_provenance": prov_omm,
  }


# ============================================================================
# S4: DHFR_24k / nb>=64 Morton 3D Cell-Grid + Octant Sub-AABB Tile Pruning
# ============================================================================
def run_suite_s4_morton_tile_pruning(
  *,
  cpu_smoke: bool = False,
  quick: bool = False,
) -> dict[str, Any]:
  """Benchmark S4: Morton cell-grid (nb >= 64) + 8-octant sub-AABB pruned tile counts & ms/step."""
  # Ensure nb >= 64 and box_a >= 3 * (r_eff + 1.0) = 31.5 A even in --cpu-smoke so the 3D cell-grid
  # path activates (704 waters = 2112 atoms -> nb = 66 >= 64, box_a = 46.0 A >= 31.5 A)
  n_waters = 704 if cpu_smoke else (2048 if quick else 7840)
  cutoff = 7.5 if cpu_smoke else 9.0
  skin = 2.0
  r_eff = cutoff + skin
  sys_s4, _, pos_nm_s4, box_a = _build_tip3p_openmm_box(
    n_waters,
    with_barostat=False,
    cutoff_nm=cutoff * 0.1,
    pme_grid=24 if cpu_smoke else 60,
    box_nm_override=4.6 if cpu_smoke else None,
  )
  conv = from_openmm_system(
    sys_s4,
    config=ProlixOpenMMConfig(precision="single", use_size_buckets=False),
    positions=pos_nm_s4 * unit.nanometer,
  )
  bun = conv.bundle
  n_real = int(jnp.sum(bun.atom_mask))
  nb = (n_real + 31) // 32
  n_act = nb * 32
  n_cur = int(bun.positions.shape[0])
  pad_len = max(0, n_act - n_cur)

  pos_raw = jnp.asarray(bun.positions[:n_act], dtype=jnp.float32)
  q_raw = jnp.asarray(bun.charges[:n_act], dtype=jnp.float32)
  sig_raw = jnp.asarray(bun.sigmas[:n_act], dtype=jnp.float32)
  eps_raw = jnp.asarray(bun.epsilons[:n_act], dtype=jnp.float32)
  mask_raw = jnp.asarray(bun.atom_mask[:n_act], dtype=bool)
  if pad_len > 0:
    pos_pad = jnp.pad(pos_raw, ((0, pad_len), (0, 0)), constant_values=0.0)
    q_pad = jnp.pad(q_raw, (0, pad_len), constant_values=0.0)
    sig_pad = jnp.pad(sig_raw, (0, pad_len), constant_values=1.0)
    eps_pad = jnp.pad(eps_raw, (0, pad_len), constant_values=0.0)
    mask_pad = jnp.pad(mask_raw, (0, pad_len), constant_values=False)
  else:
    pos_pad, q_pad, sig_pad, eps_pad, mask_pad = pos_raw, q_raw, sig_raw, eps_raw, mask_raw
  box_vec = jnp.diag(jnp.asarray(bun.box, dtype=jnp.float32))

  pos_b = pos_pad[None, :, :]
  mask_b = mask_pad[None, :]
  box_b = box_vec[None, :]
  dedup = precompute_dedup_excl_edges(
    bun.excl_dense_indices,
    bun.excl_dense_scales_vdw,
    bun.excl_dense_scales_elec,
    n_act,
  )
  max_pairs = estimate_max_tile_pairs(nb, cutoff=cutoff, skin=skin)
  all_upper_tri = (nb * (nb + 1)) // 2

  # Coarse AABB-only pair count (disable octant sub-AABB narrowing by broadcasting tile AABB)
  codes = _morton3d_encode_jax(pos_b, box_b, mask_b)
  perm = jnp.argsort(codes, axis=-1).astype(jnp.int32)
  inv_perm = jnp.argsort(perm, axis=-1).astype(jnp.int32)
  pos_m = jnp.take_along_axis(pos_b, perm[:, :, None], axis=1)
  mask_m = jnp.take_along_axis(mask_b.astype(jnp.int32), perm, axis=1)
  center, half_ext, tile_has, _, _, _ = _compute_tile_aabbs(pos_m, mask_m, box_b, nb)
  sub_c_coarse = jnp.broadcast_to(center[:, :, None, :], (1, nb, 8, 3))
  sub_h_coarse = jnp.broadcast_to(half_ext[:, :, None, :], (1, nb, 8, 3))
  sub_has_coarse = jnp.broadcast_to(tile_has[:, :, None], (1, nb, 8))
  _, _, _, _, n_val_coarse = _build_tile_pairs_jax_fallback(
    center,
    half_ext,
    tile_has,
    sub_c_coarse,
    sub_h_coarse,
    sub_has_coarse,
    box_b,
    inv_perm,
    dedup,
    (r_eff + 1e-3) ** 2,
    min(all_upper_tri, max_pairs * 2),
  )
  coarse_pairs = int(np.asarray(n_val_coarse[0]))

  # v3.6 3D cell-grid + 8-octant sub-AABB discovery
  tp = pallas_build_morton_tile_pairs_gpu(
    pos_b,
    q_pad[None, :],
    sig_pad[None, :],
    eps_pad[None, :],
    mask_b,
    box_b,
    dedup,
    n_act=n_act,
    max_pairs=max_pairs,
    cutoff=cutoff,
    skin=skin,
    host_min_box=float(jnp.min(box_vec)),
    interpret=cpu_smoke,
  )
  v36_pairs = int(np.asarray(tp.n_valid_pairs[0]))

  # Step throughput (on GPU runs full nb>=64 box; in --cpu-smoke uses compact 32-water box to keep CI < 30s)
  if cpu_smoke:
    sys_step, _, pos_nm_step, _ = _build_tip3p_openmm_box(
      32, with_barostat=False, cutoff_nm=0.40, pme_grid=16
    )
    bun_step = from_openmm_system(
      sys_step,
      config=ProlixOpenMMConfig(precision="single", use_size_buckets=False),
      positions=pos_nm_step * unit.nanometer,
    ).bundle
  else:
    sys_step, pos_nm_step, bun_step = sys_s4, pos_nm_s4, bun

  n_warmup = 1 if cpu_smoke else 15
  n_steps = 2 if cpu_smoke else (80 if quick else 200)
  plan = EnsemblePlan.from_bundle(
    bun_step,
    use_pallas=not cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nonbonded_engine="morton_tile_pairs" if not cpu_smoke else "neighbor_list",
    precision="single",
  )
  w_tr = plan.run(
    n_steps=n_warmup,
    dt=2.0,
    kT=300.0 * BOLTZMANN_KCAL_PER_MOL_K,
    gamma=1.0,
    seed=401,
    run_mode="inference",
    use_neighbor_list=True,
    nl_update_every=20,
  )
  jax.block_until_ready(w_tr.positions if hasattr(w_tr, "positions") else w_tr)
  t0 = time.perf_counter()
  tr = plan.run(
    n_steps=n_steps,
    dt=2.0,
    kT=300.0 * BOLTZMANN_KCAL_PER_MOL_K,
    gamma=1.0,
    seed=402,
    run_mode="inference",
    use_neighbor_list=True,
    nl_update_every=20,
  )
  jax.block_until_ready(tr.positions if hasattr(tr, "positions") else tr)
  prx_ms = ((time.perf_counter() - t0) / n_steps) * 1000.0

  integ_omm = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
  )
  ctx_omm, prov_omm = _openmm_context_for_mode(
    sys_step, integ_omm, expected_precision="single", cpu_smoke=cpu_smoke
  )
  ctx_omm.setPositions(pos_nm_step * unit.nanometer)
  ctx_omm.setVelocitiesToTemperature(300.0 * unit.kelvin, 402)
  integ_omm.step(n_warmup)
  _ = ctx_omm.getState(getEnergy=True)
  t_o0 = time.perf_counter()
  integ_omm.step(n_steps)
  _ = ctx_omm.getState(getEnergy=True)
  omm_ms = ((time.perf_counter() - t_o0) / n_steps) * 1000.0
  del ctx_omm, integ_omm

  pruned_vs_all_pct = 100.0 * (1.0 - float(v36_pairs) / float(max(all_upper_tri, 1)))
  pruned_vs_coarse_pct = 100.0 * (1.0 - float(v36_pairs) / float(max(coarse_pairs, 1)))
  return {
    "n_real_atoms": n_real,
    "n_tiles": nb,
    "box_angstrom": float(box_a),
    "cutoff_angstrom": float(cutoff),
    "skin_angstrom": float(skin),
    "all_upper_tri_tile_pairs": int(all_upper_tri),
    "coarse_aabb_only_tile_pairs": int(coarse_pairs),
    "v36_octant_cell_grid_tile_pairs": int(v36_pairs),
    "pruned_vs_all_pairs_pct": float(pruned_vs_all_pct),
    "pruned_vs_coarse_aabb_pct": float(pruned_vs_coarse_pct),
    "prolix_ms_per_step": float(prx_ms),
    "openmm_ms_per_step": float(omm_ms),
    "openmm_provenance": prov_omm,
  }


# ============================================================================
# S5: Pallas fltflt (hi, lo) rn.f32 Mixed/Single Ratio & Bond Conservation
# ============================================================================
def run_suite_s5_fltflt_mixed_vs_single(
  *,
  cpu_smoke: bool = False,
  quick: bool = False,
) -> dict[str, Any]:
  """Benchmark S5: fltflt mixed-precision vs single-precision ratio and SETTLE geometry error."""
  n_waters = 24 if cpu_smoke else (256 if quick else 512)
  n_warmup = 2 if cpu_smoke else 20
  n_steps = 6 if cpu_smoke else (120 if quick else 400)
  sys_s5, _, pos_nm_s5, _ = _build_tip3p_openmm_box(
    n_waters,
    with_barostat=False,
    cutoff_nm=0.75 if cpu_smoke else 0.85,
    pme_grid=16 if cpu_smoke else 32,
  )

  def _run_prec(prec: str) -> tuple[float, float, dict[str, Any] | None]:
    cfg = ProlixOpenMMConfig(
      precision=cast("Any", prec),
      use_pallas=True,
      pallas_interpret=cpu_smoke,
      use_pallas_pme=not cpu_smoke,
      nonbonded_engine="morton_tile_pairs",
      nl_update_every=4 if cpu_smoke else 20,
    )
    ctx = ProlixContext(
      sys_s5,
      mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
      ),
      config=cfg,
    )
    ctx.setPositions(pos_nm_s5 * unit.nanometer)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 505)
    ctx.step(n_warmup)
    _ = ctx.getState(getPositions=True)
    t0 = time.perf_counter()
    ctx.step(n_steps)
    st = ctx.getState(getPositions=True)
    ms_step = ((time.perf_counter() - t0) / n_steps) * 1000.0
    pos_a = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    o_p = pos_a[0::3]
    h1_p = pos_a[1::3]
    h2_p = pos_a[2::3]
    d_oh1 = np.linalg.norm(h1_p - o_p, axis=-1)
    d_oh2 = np.linalg.norm(h2_p - o_p, axis=-1)
    max_bond_err = float(
      max(np.max(np.abs(d_oh1 - TIP3P_ROH)), np.max(np.abs(d_oh2 - TIP3P_ROH)))
    )

    integ_o = mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    ctx_o, prov_o = _openmm_context_for_mode(
      sys_s5, integ_o, expected_precision=prec, cpu_smoke=cpu_smoke
    )
    ctx_o.setPositions(pos_nm_s5 * unit.nanometer)
    integ_o.step(2)
    del ctx_o, integ_o
    return ms_step, max_bond_err, prov_o

  single_ms, err_single, prov_single = _run_prec("single")
  mixed_ms, err_mixed, prov_mixed = _run_prec("mixed")
  return {
    "n_waters": n_waters,
    "n_steps": n_steps,
    "single_ms_per_step": float(single_ms),
    "mixed_fltflt_ms_per_step": float(mixed_ms),
    "mixed_over_single_ratio": float(mixed_ms / max(single_ms, 1e-12)),
    "bond_constraint_max_err_single": float(err_single),
    "bond_constraint_max_err_mixed": float(err_mixed),
    "openmm_provenance_single": prov_single,
    "openmm_provenance_mixed": prov_mixed,
  }


def run_v36_ab_benchmark(
  *,
  suites: Sequence[str] = ("S1", "S2", "S3", "S4", "S5"),
  gpu_tag: str | None = None,
  quick: bool = False,
  cpu_smoke: bool = False,
) -> dict[str, Any]:
  """Run requested v3.6/v3.7 A/B benchmark suites and return structured JSON payload."""
  t_wall0 = time.perf_counter()
  norm_suites = [s.strip().upper() for s in suites if s.strip()]
  if not cpu_smoke:
    devs = jax.devices()
    if not devs or devs[0].platform != "gpu":
      raise RuntimeError(
        f"GPU mode requires a JAX GPU backend (found {[d.platform for d in devs]}). "
        "Pass --cpu-smoke for CPU CI smoke runs."
      )

  dev0 = jax.devices()[0]
  eff_gpu_tag = gpu_tag or ("cpu_smoke" if cpu_smoke else dev0.device_kind.lower().replace(" ", "_"))
  results: dict[str, Any] = {
    "schema_version": "pallas_v36_ab_suite_v1",
    "gpu_tag": eff_gpu_tag,
    "suites_requested": ",".join(norm_suites),
    "device_kind": str(dev0.device_kind),
    "jax_version": str(jax.__version__),
    "openmm_inventory": get_openmm_platform_inventory(),
    "cpu_smoke": bool(cpu_smoke),
    "quick_mode": bool(quick),
  }

  if "S1" in norm_suites:
    results["s1_npt_tip3p_pressure_and_density"] = run_suite_s1_npt_tip3p(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S2" in norm_suites:
    results["s2_nvt_hbonds_solute_water_delta_t"] = run_suite_s2_nvt_hbonds_delta_t(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S3" in norm_suites:
    results["s3_suite_d_openmm_fastpath_and_reporters"] = run_suite_s3_openmm_fastpath_and_reporters(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S4" in norm_suites:
    results["s4_dhfr_morton_tile_pruning_and_throughput"] = run_suite_s4_morton_tile_pruning(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S5" in norm_suites:
    results["s5_fltflt_mixed_vs_single_ratio"] = run_suite_s5_fltflt_mixed_vs_single(
      cpu_smoke=cpu_smoke, quick=quick
    )

  results["total_wall_s"] = float(time.perf_counter() - t_wall0)
  return results


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Turnkey Pallas v3.6/v3.7 A/B Physics & OpenMM Fast-Path Benchmark Suite."
  )
  parser.add_argument(
    "--out",
    "--output",
    dest="out",
    type=Path,
    default=Path("tmp/bench_v36_cupotential_ab.json"),
    help="Output JSON path (compatible with bathos `bth run --output-paths`).",
  )
  parser.add_argument(
    "--gpu-tag",
    type=str,
    default=None,
    help="Hardware tag recorded in output JSON (e.g., l40s, a100, h200, blackwell).",
  )
  parser.add_argument(
    "--suites",
    type=str,
    default="S1,S2,S3,S4,S5",
    help="Comma-separated subset of suites to run (S1,S2,S3,S4,S5).",
  )
  parser.add_argument(
    "--quick",
    action="store_true",
    help="Use reduced trajectory lengths for fast GPU gate verification.",
  )
  parser.add_argument(
    "--cpu-smoke",
    action="store_true",
    help="Run deterministic CPU interpret smoke verification for CI (disables CUDA guard).",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  suite_list = [s.strip() for s in args.suites.split(",") if s.strip()]
  payload = run_v36_ab_benchmark(
    suites=suite_list,
    gpu_tag=args.gpu_tag,
    quick=args.quick,
    cpu_smoke=args.cpu_smoke,
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
