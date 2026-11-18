# ruff: noqa: E402
"""Turnkey Pallas v3.9 Triclinic, Prolate Splitting, Alchemical & DLPack Benchmark Suite.

Evaluates five Pallas v3.9 suites side-by-side against guarded OpenMM baselines:
  - S1: Reduced-form lower-triangular triclinic explicit solvent (rhombic dodecahedron
        and truncated octahedron) Morton tile pairs + SPME + BAOAB SETTLE parity,
        step latency, and holonomic bond conservation.
  - S2: Cross-shape 1D ragged batching mixing cubic, rhombic-dodecahedron, and
        truncated-octahedron replicas with deterministic (hi_i32, lo_i32) forces.
  - S3: Clean-room C^2-continuous Prolate Spheroidal Ewald splittings (arXiv:2505.09727)
        for P=1 electrostatics (c_1 = 10.75) and P=6 dispersion (c_6 = 12.764031957789484):
        C^2 cutoff vanishing, k=0 dispersion limit, autodiff forces, and periodic parity.
  - S4: Sparse O(|U|*N) CustomNonbondedForce InteractionGroups (is_gpu_runner_fast) and
        zero-recompilation multi-window alchemical lambda sweeps + dU/dlambda derivatives.
  - S5: GPU-resident MonteCarloAnisotropicBarostat & MonteCarloMembraneBarostat, 4.0 fs
        Hydrogen Mass Repartitioning (HMR), and zero-copy JAX/DLPack state access.

On GPU runs (default), OpenMM baselines strictly enforce `require_openmm_cuda_context`
(`libOpenMMCUDA.so` mapped, zero silent OpenCL/CPU/Reference fallback). Pass
`--cpu-smoke` for deterministic CI verification without GPU hardware.
"""

from __future__ import annotations

import argparse
import inspect
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

from prolix.openmm import (
  ProlixBatchSimulation,
  ProlixContext,
  ProlixOpenMMConfig,
  ProlixSimulation,
)
from prolix.openmm.lepton_compiler import compile_custom_force
from prolix.physics import prolate_splitting as ps_mod
from prolix.physics.pbc import minimum_image_distance, triclinic_box_from_lengths_angles
from prolix.physics.settle import TIP3P_RHH, TIP3P_ROH
from scripts.benchmarks.openmm_cuda_guard import (
  extract_openmm_context_provenance,
  get_openmm_platform_inventory,
  require_openmm_cuda_context,
)

if TYPE_CHECKING:
  from collections.abc import Sequence

unit = cast("Any", _omm_unit)

KCAL_TO_KJ = 4.184
TIP3P_Q_O = -0.834
TIP3P_Q_H = 0.417
TIP3P_SIG_O_NM = 0.3150752406575124
TIP3P_EPS_O_KJ = 0.635968


class _DLPackTensorView:
  """Minimal third-party DLPack producer wrapper verifying `from_dlpack` ingestion."""

  def __init__(self, arr: jax.Array) -> None:
    self._arr = arr

  def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
    return cast("Any", self._arr).__dlpack__(*args, **kwargs)

  def __dlpack_device__(self) -> tuple[int, int]:
    return cast("Any", self._arr).__dlpack_device__()


def _openmm_context_for_mode(
  system: mm.System,
  integrator: mm.Integrator,
  *,
  expected_precision: str = "single",
  cpu_smoke: bool = False,
  force_reference: bool = False,
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
  if force_reference:
    plat_name = "Reference"
  else:
    plat_name = "CPU" if "CPU" in inv["platforms"] else "Reference"
  plat = mm.Platform.getPlatformByName(plat_name)
  ctx = mm.Context(system, integrator, plat)
  return ctx, extract_openmm_context_provenance(ctx)


def _make_box_matrix_nm(shape_kind: str, edge_nm: float) -> np.ndarray:
  """Return OpenMM reduced-form lower-triangular `(3, 3)` lattice matrix in nanometers."""
  edge_a = float(edge_nm) * 10.0
  if shape_kind == "cubic":
    return np.diag([edge_nm, edge_nm, edge_nm]).astype(np.float64)
  if shape_kind == "rhombic_dodecahedron":
    box_a = np.asarray(
      triclinic_box_from_lengths_angles(
        edge_a, edge_a, edge_a, 60.0, 60.0, 60.0, reduce=True, dtype=jnp.float64
      ),
      dtype=np.float64,
    )
    return box_a * 0.1
  if shape_kind == "truncated_octahedron":
    oct_deg = math.degrees(math.acos(-1.0 / 3.0))
    box_a = np.asarray(
      triclinic_box_from_lengths_angles(
        edge_a, edge_a, edge_a, oct_deg, oct_deg, oct_deg, reduce=True, dtype=jnp.float64
      ),
      dtype=np.float64,
    )
    return box_a * 0.1
  raise ValueError(f"Unsupported box shape_kind={shape_kind!r}")


def _build_water_box_system(
  n_waters: int,
  *,
  shape_kind: str = "cubic",
  edge_nm: float = 2.60,
  cutoff_nm: float = 0.80,
  pme_alpha_nm_inv: float = 3.5,
  pme_grid: int = 24,
  hmr_hydrogen_mass_amu: float | None = None,
  frac_span: float = 1.0,
  seed: int = 20261118,
) -> tuple[mm.System, app.Topology, np.ndarray, np.ndarray]:
  """Construct an orthorhombic or reduced-triclinic rigid-TIP3P OpenMM System & positions (nm)."""
  box_nm = _make_box_matrix_nm(shape_kind, edge_nm)
  min_diag_nm = float(min(box_nm[0, 0], box_nm[1, 1], box_nm[2, 2]))
  if min_diag_nm <= 2.0 * cutoff_nm + 0.05:
    scale = (2.0 * cutoff_nm + 0.25) / min_diag_nm
    box_nm = box_nm * scale

  top = app.Topology()
  chain = top.addChain()
  elem_o = app.Element.getBySymbol("O")
  elem_h = app.Element.getBySymbol("H")

  sys_mm = mm.System()
  sys_mm.setDefaultPeriodicBoxVectors(
    mm.Vec3(float(box_nm[0, 0]), float(box_nm[0, 1]), float(box_nm[0, 2])),
    mm.Vec3(float(box_nm[1, 0]), float(box_nm[1, 1]), float(box_nm[1, 2])),
    mm.Vec3(float(box_nm[2, 0]), float(box_nm[2, 1]), float(box_nm[2, 2])),
  )

  nb = mm.NonbondedForce()
  nb.setNonbondedMethod(mm.NonbondedForce.PME)
  nb.setCutoffDistance(cutoff_nm * unit.nanometer)
  nb.setPMEParameters(pme_alpha_nm_inv, pme_grid, pme_grid, pme_grid)
  nb.setUseDispersionCorrection(False)

  if hmr_hydrogen_mass_amu is not None:
    m_h = float(hmr_hydrogen_mass_amu)
    m_o = 18.015 - 2.0 * m_h
  else:
    m_h = 1.008
    m_o = 15.999

  n_side = max(2, math.ceil(n_waters ** (1.0 / 3.0)))
  rng = np.random.default_rng(seed)
  theta = math.radians(104.52)
  r_oh_nm = TIP3P_ROH * 0.1
  r_hh_nm = TIP3P_RHH * 0.1
  h1_off = np.array([r_oh_nm, 0.0, 0.0], dtype=np.float64)
  h2_off = np.array([r_oh_nm * math.cos(theta), r_oh_nm * math.sin(theta), 0.0], dtype=np.float64)

  coords_nm: list[np.ndarray] = []
  s_off = 0.5 * (1.0 - float(frac_span))
  placed = 0
  for ix in range(n_side):
    for iy in range(n_side):
      for iz in range(n_side):
        if placed >= n_waters:
          break
        s_frac = s_off + float(frac_span) * np.array(
          [
            (ix + 0.5) / float(n_side),
            (iy + 0.5) / float(n_side),
            (iz + 0.5) / float(n_side),
          ],
          dtype=np.float64,
        )
        jitter = rng.uniform(-0.005, 0.005, size=3)
        dip_sign = -1.0 if ((ix + iy + iz) % 2 == 1) else 1.0
        o_pos = s_frac @ box_nm + jitter
        h1_pos = o_pos + dip_sign * h1_off
        h2_pos = o_pos + dip_sign * h2_off
        coords_nm.extend([o_pos, h1_pos, h2_pos])

        res = top.addResidue("HOH", chain)
        ao = top.addAtom("O", elem_o, res)
        ah1 = top.addAtom("H1", elem_h, res)
        ah2 = top.addAtom("H2", elem_h, res)
        top.addBond(ao, ah1)
        top.addBond(ao, ah2)

        idx_o = sys_mm.addParticle(m_o * unit.amu)
        idx_h1 = sys_mm.addParticle(m_h * unit.amu)
        idx_h2 = sys_mm.addParticle(m_h * unit.amu)

        nb.addParticle(TIP3P_Q_O, TIP3P_SIG_O_NM, TIP3P_EPS_O_KJ)
        nb.addParticle(TIP3P_Q_H, 0.1, 0.0)
        nb.addParticle(TIP3P_Q_H, 0.1, 0.0)

        nb.addException(idx_o, idx_h1, 0.0, 0.1, 0.0)
        nb.addException(idx_o, idx_h2, 0.0, 0.1, 0.0)
        nb.addException(idx_h1, idx_h2, 0.0, 0.1, 0.0)

        sys_mm.addConstraint(idx_o, idx_h1, r_oh_nm * unit.nanometer)
        sys_mm.addConstraint(idx_o, idx_h2, r_oh_nm * unit.nanometer)
        sys_mm.addConstraint(idx_h1, idx_h2, r_hh_nm * unit.nanometer)
        placed += 1

  sys_mm.addForce(nb)
  return sys_mm, top, np.asarray(coords_nm, dtype=np.float64), box_nm


def _max_triclinic_water_bond_err_a(pos_a: np.ndarray, box_a: np.ndarray) -> float:
  """Compute max |d_OH - r_OH| and |d_HH - r_HH| in Angstroms under triclinic MIC."""
  pos_j = jnp.asarray(pos_a, dtype=jnp.float64)
  box_j = jnp.asarray(box_a, dtype=jnp.float64)
  o_p = pos_j[0::3]
  h1_p = pos_j[1::3]
  h2_p = pos_j[2::3]
  d_oh1 = np.asarray(minimum_image_distance(h1_p, o_p, box_j))
  d_oh2 = np.asarray(minimum_image_distance(h2_p, o_p, box_j))
  d_hh = np.asarray(minimum_image_distance(h2_p, h1_p, box_j))
  return float(
    max(
      np.max(np.abs(d_oh1 - TIP3P_ROH)),
      np.max(np.abs(d_oh2 - TIP3P_ROH)),
      np.max(np.abs(d_hh - TIP3P_RHH)),
    )
  )


def run_suite_s1_triclinic_explicit(*, cpu_smoke: bool = False, quick: bool = False) -> dict[str, Any]:
  """S1: Reduced-form rhombic dodecahedron & truncated octahedron parity and step latency."""
  n_waters = 16 if cpu_smoke else (64 if quick else 256)
  edge_nm = 2.55 if cpu_smoke else (2.90 if quick else 3.80)
  n_warmup = 1 if cpu_smoke else 10
  n_steps = 2 if cpu_smoke else (25 if quick else 100)

  sub_results: dict[str, Any] = {}
  last_prov: dict[str, Any] = {}

  for shape_kind in ("rhombic_dodecahedron", "truncated_octahedron"):
    sys_mm, _top, pos_nm, box_nm = _build_water_box_system(
      n_waters,
      shape_kind=shape_kind,
      edge_nm=edge_nm,
      cutoff_nm=0.80,
      pme_alpha_nm_inv=3.5,
      pme_grid=48,
      frac_span=0.58 if cpu_smoke else 1.0,
      seed=1101 if shape_kind == "rhombic_dodecahedron" else 1102,
    )
    cfg = ProlixOpenMMConfig(
      precision="mixed",
      nonbonded_engine="morton_tile_pairs",
      use_pallas=not cpu_smoke,
      use_pallas_pme=not cpu_smoke,
      nl_update_every=4 if cpu_smoke else 10,
    )
    ctx_p = ProlixContext(
      sys_mm,
      mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
      config=cfg,
    )
    ctx_p.setPositions(pos_nm * unit.nanometer)
    ctx_p.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)

    st0_p = ctx_p.getState(getEnergy=True, getForces=True)
    e_prolix_kj = float(st0_p.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    f_prolix = np.asarray(
      st0_p.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
      dtype=np.float64,
    )

    integ_o = mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    ctx_o, prov_o = _openmm_context_for_mode(
      sys_mm, integ_o, expected_precision="mixed", cpu_smoke=cpu_smoke
    )
    last_prov = prov_o
    ctx_o.setPositions(pos_nm * unit.nanometer)
    ctx_o.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
    st0_o = ctx_o.getState(getEnergy=True, getForces=True)
    e_openmm_kj = float(st0_o.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    f_openmm = np.asarray(
      st0_o.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
      dtype=np.float64,
    )

    rel_energy_err = abs(e_prolix_kj - e_openmm_kj) / max(abs(e_openmm_kj), 1.0)
    f_denom = max(float(np.max(np.linalg.norm(f_openmm, axis=-1))), 1.0)
    max_force_rel_err = float(np.max(np.linalg.norm(f_prolix - f_openmm, axis=-1)) / f_denom)

    ctx_p.step(n_warmup)
    _ = ctx_p.getState(getPositions=True)
    t0 = time.perf_counter()
    ctx_p.step(n_steps)
    st_end = ctx_p.getState(getPositions=True)
    prolix_ms = ((time.perf_counter() - t0) / n_steps) * 1000.0

    pos_end_a = np.asarray(st_end.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    bond_err_a = _max_triclinic_water_bond_err_a(pos_end_a, box_nm * 10.0)

    integ_o.step(n_warmup)
    _ = ctx_o.getState(getPositions=True)
    t1 = time.perf_counter()
    integ_o.step(n_steps)
    _ = ctx_o.getState(getPositions=True)
    openmm_ms = ((time.perf_counter() - t1) / n_steps) * 1000.0
    del ctx_o, integ_o

    sub_results[shape_kind] = {
      "n_waters": n_waters,
      "n_atoms": 3 * n_waters,
      "n_steps": n_steps,
      "morton_fastpath_active": bool(getattr(ctx_p, "_morton_runner", None) is not None),
      "e_prolix_kj_mol": e_prolix_kj,
      "e_openmm_kj_mol": e_openmm_kj,
      "rel_energy_err": float(rel_energy_err),
      "max_force_rel_err": float(max_force_rel_err),
      "prolix_ms_per_step": float(prolix_ms),
      "openmm_ms_per_step": float(openmm_ms),
      "max_holonomic_bond_err_a": float(bond_err_a),
      "openmm_provenance": prov_o,
    }

  dodec = sub_results["rhombic_dodecahedron"]
  octa = sub_results["truncated_octahedron"]
  return {
    "rhombic_dodecahedron": dodec,
    "truncated_octahedron": octa,
    "max_rel_energy_err": float(max(dodec["rel_energy_err"], octa["rel_energy_err"])),
    "max_force_rel_err": float(max(dodec["max_force_rel_err"], octa["max_force_rel_err"])),
    "max_holonomic_bond_err_a": float(
      max(dodec["max_holonomic_bond_err_a"], octa["max_holonomic_bond_err_a"])
    ),
    "openmm_provenance": last_prov,
  }


def run_suite_s2_ragged_cross_shape_batch(
  *, cpu_smoke: bool = False, quick: bool = False
) -> dict[str, Any]:
  """S2: Cross-size 1D ragged batching mixing cubic, dodecahedron, and octahedron replicas."""
  specs = (
    [
      ("cubic", 12, 2.15),
      ("rhombic_dodecahedron", 16, 2.55),
      ("truncated_octahedron", 20, 2.60),
    ]
    if cpu_smoke
    else (
      [
        ("cubic", 64, 2.80),
        ("rhombic_dodecahedron", 96, 3.10),
        ("truncated_octahedron", 128, 3.35),
      ]
      if quick
      else [
        ("cubic", 128, 3.30),
        ("rhombic_dodecahedron", 192, 3.70),
        ("truncated_octahedron", 256, 4.00),
      ]
    )
  )
  n_warmup = 1 if cpu_smoke else 8
  n_steps = 2 if cpu_smoke else (20 if quick else 80)

  systems: list[mm.System] = []
  topologies: list[app.Topology] = []
  positions_list: list[Any] = []
  for idx, (shape_kind, nw, edge_nm) in enumerate(specs):
    sys_i, top_i, pos_i, _ = _build_water_box_system(
      nw,
      shape_kind=shape_kind,
      edge_nm=edge_nm,
      cutoff_nm=0.80,
      pme_alpha_nm_inv=3.5,
      pme_grid=16 if cpu_smoke else 24,
      seed=2200 + idx,
    )
    systems.append(sys_i)
    topologies.append(top_i)
    positions_list.append(pos_i * unit.nanometer)

  cfg = ProlixOpenMMConfig(
    precision="mixed",
    nonbonded_engine="morton_tile_pairs" if not cpu_smoke else "neighbor_list",
    deterministic_forces=True,
    use_pallas=not cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nl_update_every=4 if cpu_smoke else 10,
  )
  integ_tmpl = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
  )
  batch = ProlixBatchSimulation(
    systems=systems,
    integrators=integ_tmpl,
    topologies=topologies,
    positions_list=positions_list,
    config=cfg,
  )
  batch.setVelocitiesToTemperature(300.0 * unit.kelvin, 2026)

  f_rep_a = np.asarray(batch.getForcesJax(in_kj_per_mol_nm=True))
  f_rep_b = np.asarray(batch.getForcesJax(in_kj_per_mol_nm=True))
  det_max_diff = float(np.max(np.abs(f_rep_a - f_rep_b)))

  batch.step(n_warmup)
  _ = batch.getStates(getEnergy=True)
  t0 = time.perf_counter()
  batch.step(n_steps)
  states = batch.getStates(getEnergy=True)
  batch_ms = ((time.perf_counter() - t0) / n_steps) * 1000.0

  energies_kj = [
    float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)) for st in states
  ]

  integ_o = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
  )
  ctx_o, prov_o = _openmm_context_for_mode(
    systems[-1], integ_o, expected_precision="mixed", cpu_smoke=cpu_smoke
  )
  ctx_o.setPositions(positions_list[-1])
  integ_o.step(2)
  del ctx_o, integ_o

  return {
    "batch_size": int(batch.batch_size),
    "replica_shapes": [s[0] for s in specs],
    "replica_atom_counts": [3 * s[1] for s in specs],
    "n_steps": n_steps,
    "deterministic_force_max_diff": det_max_diff,
    "prolix_batch_ms_per_step": float(batch_ms),
    "per_replica_potential_energies_kj_mol": energies_kj,
    "openmm_provenance": prov_o,
  }


def _invoke_prolate_spme_electrostatics(
  positions_a: jax.Array,
  charges: jax.Array,
  box_a: jax.Array,
  cutoff_a: float,
  grid_dims: tuple[int, int, int],
) -> jax.Array:
  """Call `prolate_spme_electrostatics` binding exact signature parameter names."""
  fn = ps_mod.prolate_spme_electrostatics
  sig = inspect.signature(fn)
  params = sig.parameters
  kwargs: dict[str, Any] = {}
  if "cutoff" in params:
    kwargs["cutoff"] = cutoff_a
  elif "rc" in params:
    kwargs["rc"] = cutoff_a
  elif "r_cut" in params:
    kwargs["r_cut"] = cutoff_a
  if "grid_dims" in params:
    kwargs["grid_dims"] = grid_dims
  if "neighbor_idx" in params:
    n = int(positions_a.shape[0])
    all_idx = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32)[None, :], (n, n))
    self_col = jnp.arange(n, dtype=jnp.int32)[:, None]
    kwargs["neighbor_idx"] = jnp.where(all_idx == self_col, -1, all_idx)
  return fn(positions_a, charges, box_a, **kwargs)


def _invoke_prolate_dispersion_pme(
  positions_a: jax.Array,
  sigmas_a: jax.Array,
  epsilons_kcal: jax.Array,
  box_a: jax.Array,
  cutoff_a: float,
  grid_dims: tuple[int, int, int],
) -> jax.Array:
  """Call `prolate_dispersion_pme` binding exact signature parameter names."""
  fn = ps_mod.prolate_dispersion_pme
  sig = inspect.signature(fn)
  params = sig.parameters
  pos_names = [
    name
    for name, p in params.items()
    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
  ]
  kwargs: dict[str, Any] = {}
  if "cutoff" in params:
    kwargs["cutoff"] = cutoff_a
  elif "rc" in params:
    kwargs["rc"] = cutoff_a
  elif "r_cut" in params:
    kwargs["r_cut"] = cutoff_a
  if "grid_dims" in params:
    kwargs["grid_dims"] = grid_dims
  if "neighbor_idx" in params:
    n = int(positions_a.shape[0])
    all_idx = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32)[None, :], (n, n))
    self_col = jnp.arange(n, dtype=jnp.int32)[:, None]
    kwargs["neighbor_idx"] = jnp.where(all_idx == self_col, -1, all_idx)
  if "include_repulsion" in params:
    kwargs["include_repulsion"] = False

  # Some dispersion solvers accept (positions, c6_or_b, box) vs (positions, sigmas, epsilons, box)
  if len(pos_names) >= 3 and pos_names[1] in ("b_disp", "c6_weights", "dispersion_charges", "b"):
    b_disp = 2.0 * jnp.sqrt(jnp.maximum(epsilons_kcal, 0.0)) * (sigmas_a**3)
    return fn(positions_a, b_disp, box_a, **kwargs)
  return fn(positions_a, sigmas_a, epsilons_kcal, box_a, **kwargs)


def run_suite_s3_prolate_c2_splitting(
  *, cpu_smoke: bool = False, quick: bool = False
) -> dict[str, Any]:
  """S3: Clean-room C^2 Prolate Spheroidal Ewald splittings (P=1 & P=6, arXiv:2505.09727)."""
  rc_a = 9.0
  rc2 = jnp.asarray(rc_a * rc_a, dtype=jnp.float64)

  p1_s_rc, p1_ds_rc = ps_mod.prolate_p1_short_range_value_and_d_dr2(rc2, rc_a)
  p6_s_rc, p6_ds_rc = ps_mod.prolate_p6_short_range_value_and_d_dr2(rc2, rc_a)
  p6_k0_val, p6_k0_grad = ps_mod.prolate_p6_fourier_kernel_and_d_dk2(
    jnp.asarray(0.0, dtype=jnp.float64), rc_a
  )

  n_particles = 24 if cpu_smoke else (64 if quick else 128)
  box_a_scalar = 24.0 if cpu_smoke else 28.0
  grid_dims = (24, 24, 24) if cpu_smoke else (32, 32, 32)
  rng = np.random.default_rng(3301)

  n_side = max(2, math.ceil(n_particles ** (1.0 / 3.0)))
  pts: list[np.ndarray] = []
  for ix in range(n_side):
    for iy in range(n_side):
      for iz in range(n_side):
        if len(pts) >= n_particles:
          break
        base = (np.array([ix + 0.5, iy + 0.5, iz + 0.5]) / float(n_side)) * box_a_scalar
        pts.append(base + rng.uniform(-0.15, 0.15, size=3))
  pos_a = jnp.asarray(np.asarray(pts, dtype=np.float64), dtype=jnp.float64)
  box_a = jnp.asarray([box_a_scalar, box_a_scalar, box_a_scalar], dtype=jnp.float64)

  charges_np = np.where(np.arange(n_particles) % 2 == 0, 0.5, -0.5).astype(np.float64)
  charges_j = jnp.asarray(charges_np, dtype=jnp.float64)
  sigmas_a = jnp.full((n_particles,), 3.2, dtype=jnp.float64)
  epsilons_kcal = jnp.full((n_particles,), 0.15, dtype=jnp.float64)

  p1_val_and_grad = jax.jit(
    jax.value_and_grad(
      lambda r: _invoke_prolate_spme_electrostatics(r, charges_j, box_a, rc_a, grid_dims)
    )
  )
  p6_val_and_grad = jax.jit(
    jax.value_and_grad(
      lambda r: _invoke_prolate_dispersion_pme(
        r, sigmas_a, epsilons_kcal, box_a, rc_a, grid_dims
      )
    )
  )

  e_p1_kcal, g_p1 = p1_val_and_grad(pos_a)
  jax.block_until_ready(g_p1)
  t0 = time.perf_counter()
  e_p1_kcal, g_p1 = p1_val_and_grad(pos_a)
  jax.block_until_ready(g_p1)
  p1_ms = (time.perf_counter() - t0) * 1000.0

  e_p6_kcal, g_p6 = p6_val_and_grad(pos_a)
  jax.block_until_ready(g_p6)
  t1 = time.perf_counter()
  e_p6_kcal, g_p6 = p6_val_and_grad(pos_a)
  jax.block_until_ready(g_p6)
  p6_ms = (time.perf_counter() - t1) * 1000.0

  # Guarded OpenMM PME electrostatic baseline (zero LJ so pure P=1 Coulomb is compared)
  sys_elec = mm.System()
  box_nm_s = box_a_scalar * 0.1
  sys_elec.setDefaultPeriodicBoxVectors(
    mm.Vec3(box_nm_s, 0.0, 0.0),
    mm.Vec3(0.0, box_nm_s, 0.0),
    mm.Vec3(0.0, 0.0, box_nm_s),
  )
  nb_elec = mm.NonbondedForce()
  nb_elec.setNonbondedMethod(mm.NonbondedForce.PME)
  nb_elec.setCutoffDistance((rc_a * 0.1) * unit.nanometer)
  nb_elec.setEwaldErrorTolerance(1.0e-5)
  for q_val in charges_np:
    sys_elec.addParticle(39.948 * unit.amu)
    nb_elec.addParticle(float(q_val), 0.32, 0.0)
  sys_elec.addForce(nb_elec)

  integ_o = mm.VerletIntegrator(0.001 * unit.picoseconds)
  ctx_o, prov_o = _openmm_context_for_mode(
    sys_elec, integ_o, expected_precision="single", cpu_smoke=cpu_smoke
  )
  ctx_o.setPositions((np.asarray(pos_a) * 0.1) * unit.nanometer)
  e_omm_elec_kcal = (
    float(ctx_o.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    / KCAL_TO_KJ
  )
  del ctx_o, integ_o

  p1_rel_err = abs(float(e_p1_kcal) - e_omm_elec_kcal) / max(abs(e_omm_elec_kcal), 1.0)

  return {
    "n_particles": n_particles,
    "cutoff_a": rc_a,
    "grid_dims": list(grid_dims),
    "p1_val_at_rc": float(abs(p1_s_rc)),
    "p1_d_dr2_at_rc": float(abs(p1_ds_rc)),
    "p6_val_at_rc": float(abs(p6_s_rc)),
    "p6_d_dr2_at_rc": float(abs(p6_ds_rc)),
    "p6_k0_val": float(p6_k0_val),
    "p6_k0_grad": float(p6_k0_grad),
    "prolate_p1_kcal_mol": float(e_p1_kcal),
    "openmm_pme_kcal_mol": float(e_omm_elec_kcal),
    "p1_rel_energy_err": float(p1_rel_err),
    "prolate_p6_kcal_mol": float(e_p6_kcal),
    "p1_grad_finite": bool(np.all(np.isfinite(np.asarray(g_p1)))),
    "p6_grad_finite": bool(np.all(np.isfinite(np.asarray(g_p6)))),
    "prolate_p1_ms": float(p1_ms),
    "prolate_p6_ms": float(p6_ms),
    "openmm_provenance": prov_o,
  }


def run_suite_s4_sparse_interaction_groups_alchemical(
  *, cpu_smoke: bool = False, quick: bool = False
) -> dict[str, Any]:
  """S4: Sparse O(|U|*N) CustomNonbondedForce InteractionGroups + dynamic alchemical lambda sweep."""
  n_waters = 16 if cpu_smoke else (96 if quick else 512)
  edge_nm = 2.20 if cpu_smoke else (3.10 if quick else 5.00)
  n_steps = 1 if cpu_smoke else (15 if quick else 50)

  sys_mm, _top, water_pos_nm, box_nm = _build_water_box_system(
    n_waters,
    shape_kind="cubic",
    edge_nm=edge_nm,
    cutoff_nm=0.80,
    pme_alpha_nm_inv=3.5,
    pme_grid=24,
    seed=4401,
  )

  # Add a 6-atom neutral ligand cluster coupled to solvent via sparse InteractionGroup
  nb_std = cast("mm.NonbondedForce", sys_mm.getForce(0))
  n_env = sys_mm.getNumParticles()
  lig_indices: list[int] = []
  lig_coords_nm: list[np.ndarray] = []
  center_nm = np.array([0.5 * box_nm[0, 0], 0.5 * box_nm[1, 1], 0.5 * box_nm[2, 2]])
  bond_f = mm.HarmonicBondForce()
  for k in range(6):
    angle = 2.0 * math.pi * float(k) / 6.0
    p_nm = center_nm + np.array([0.14 * math.cos(angle), 0.14 * math.sin(angle), 0.08])
    lig_coords_nm.append(p_nm)
    p_idx = sys_mm.addParticle(12.011 * unit.amu)
    lig_indices.append(p_idx)
    nb_std.addParticle(0.0, 0.34, 0.0)
  for k in range(6):
    bond_f.addBond(lig_indices[k], lig_indices[(k + 1) % 6], 0.14 * unit.nanometer, 250000.0)
    nb_std.addException(lig_indices[k], lig_indices[(k + 1) % 6], 0.0, 0.1, 0.0)
  sys_mm.addForce(bond_f)

  cnb = mm.CustomNonbondedForce(
    "lambda_sterics * 4.0 * eps * ((sig / r_eff)^12 - (sig / r_eff)^6);"
    "r_eff = sqrt(r^2 + 0.04);"
    "sig = 0.5 * (sig1 + sig2);"
    "eps = sqrt(eps1 * eps2)"
  )
  cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  cnb.setCutoffDistance(0.80 * unit.nanometer)
  cnb.addGlobalParameter("lambda_sterics", 0.5)
  cnb.addPerParticleParameter("sig")
  cnb.addPerParticleParameter("eps")
  for i in range(n_env):
    is_o = i % 3 == 0
    cnb.addParticle([0.315 if is_o else 0.10, 0.636 if is_o else 0.0])
  for _ in lig_indices:
    cnb.addParticle([0.34, 0.45])
  for ex_idx in range(nb_std.getNumExceptions()):
    p1, p2, *_ = nb_std.getExceptionParameters(ex_idx)
    cnb.addExclusion(int(p1), int(p2))
  cnb.addInteractionGroup(set(lig_indices), set(range(n_env)))
  cnb.addEnergyParameterDerivative("lambda_sterics")
  sys_mm.addForce(cnb)

  all_pos_nm = np.vstack([water_pos_nm, np.asarray(lig_coords_nm, dtype=np.float64)])

  compiled_cnb = compile_custom_force(cnb, n_particles=sys_mm.getNumParticles())
  is_fast = bool(getattr(compiled_cnb, "is_gpu_runner_fast", False))

  cfg = ProlixOpenMMConfig(
    precision="mixed",
    nonbonded_engine="morton_tile_pairs" if not cpu_smoke else "neighbor_list",
    use_pallas=not cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nl_update_every=4 if cpu_smoke else 10,
  )
  ctx_p = ProlixContext(
    sys_mm,
    mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds),
    config=cfg,
  )
  ctx_p.setPositions(all_pos_nm * unit.nanometer)
  ctx_p.setParameter("lambda_sterics", 0.5)
  e_p_kj = float(ctx_p.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

  integ_o = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
  )
  ctx_o, prov_o = _openmm_context_for_mode(
    sys_mm, integ_o, expected_precision="mixed", cpu_smoke=cpu_smoke
  )
  ctx_o.setPositions(all_pos_nm * unit.nanometer)
  ctx_o.setParameter("lambda_sterics", 0.5)
  e_o_kj = float(ctx_o.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
  del ctx_o, integ_o

  sparse_ig_rel_err = abs(e_p_kj - e_o_kj) / max(abs(e_o_kj), 1.0)

  lambdas = [0.0, 1.0] if cpu_smoke else [0.0, 0.25, 0.50, 0.75, 1.0]
  batch = ProlixBatchSimulation(
    systems=[sys_mm] * len(lambdas),
    integrators=mm.LangevinMiddleIntegrator(
      300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
    ),
    positions_list=[all_pos_nm * unit.nanometer] * len(lambdas),
    config=cfg,
  )
  batch.setVelocitiesToTemperature(300.0 * unit.kelvin, 4402)
  batch.setParameter("lambda_sterics", lambdas)
  t0 = time.perf_counter()
  batch.step(n_steps)
  states = batch.getStates(getEnergy=True)
  alchem_ms = ((time.perf_counter() - t0) / n_steps) * 1000.0
  derivs = batch.compute_parameter_derivatives()

  win_energies = [
    float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)) for st in states
  ]
  du_dlam = [float(d["lambda_sterics"]) for d in derivs]

  return {
    "n_total_atoms": int(sys_mm.getNumParticles()),
    "n_ligand_atoms": len(lig_indices),
    "is_gpu_runner_fast": is_fast,
    "sparse_exclusion_omitted": bool(getattr(compiled_cnb, "exclusion_mask", None) is None),
    "e_prolix_kj_mol": e_p_kj,
    "e_openmm_kj_mol": e_o_kj,
    "sparse_ig_rel_err": float(sparse_ig_rel_err),
    "lambda_windows": lambdas,
    "window_potential_energies_kj_mol": win_energies,
    "du_dlambda_per_window": du_dlam,
    "alchem_batch_ms_per_step": float(alchem_ms),
    "openmm_provenance": prov_o,
  }


def run_suite_s5_barostats_hmr_and_dlpack(
  *, cpu_smoke: bool = False, quick: bool = False
) -> dict[str, Any]:
  """S5: GPU anisotropic/membrane barostats, 4.0 fs HMR, and zero-copy JAX/DLPack access."""
  n_waters = 16 if cpu_smoke else (96 if quick else 256)
  edge_nm = 2.15 if cpu_smoke else (3.10 if quick else 3.80)
  n_steps = 2 if cpu_smoke else (30 if quick else 100)
  grid_s5 = 16 if cpu_smoke else 24

  cfg = ProlixOpenMMConfig(
    precision="mixed",
    nonbonded_engine="morton_tile_pairs" if not cpu_smoke else "neighbor_list",
    use_pallas=not cpu_smoke,
    use_pallas_pme=not cpu_smoke,
    nl_update_every=4 if cpu_smoke else 10,
  )

  # 1. MonteCarloAnisotropicBarostat
  sys_aniso, _top_a, pos_nm, _ = _build_water_box_system(
    n_waters, shape_kind="cubic", edge_nm=edge_nm, cutoff_nm=0.80, pme_grid=grid_s5, seed=5501
  )
  sys_aniso.addForce(
    mm.MonteCarloAnisotropicBarostat(
      mm.Vec3(1.0, 1.0, 1.0) * unit.bar, 300.0 * unit.kelvin, True, True, True, 2
    )
  )
  ctx_aniso = ProlixContext(
    sys_aniso,
    mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
    config=cfg,
  )
  ctx_aniso.setPositions(pos_nm * unit.nanometer)
  ctx_aniso.setVelocitiesToTemperature(300.0 * unit.kelvin, 5511)
  ctx_aniso.step(2)
  t0 = time.perf_counter()
  ctx_aniso.step(n_steps)
  st_aniso = ctx_aniso.getState(getPositions=True)
  aniso_ms = ((time.perf_counter() - t0) / n_steps) * 1000.0
  aniso_vol_nm3 = float(st_aniso.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3))

  # 2. MonteCarloMembraneBarostat
  sys_mem, _top_m, pos_mem_nm, _ = _build_water_box_system(
    n_waters, shape_kind="cubic", edge_nm=edge_nm, cutoff_nm=0.80, pme_grid=grid_s5, seed=5502
  )
  sys_mem.addForce(
    mm.MonteCarloMembraneBarostat(
      1.0 * unit.bar,
      0.0 * unit.bar * unit.nanometer,
      300.0 * unit.kelvin,
      mm.MonteCarloMembraneBarostat.XYIsotropic,
      mm.MonteCarloMembraneBarostat.ZFree,
      2,
    )
  )
  ctx_mem = ProlixContext(
    sys_mem,
    mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
    config=cfg,
  )
  ctx_mem.setPositions(pos_mem_nm * unit.nanometer)
  ctx_mem.setVelocitiesToTemperature(300.0 * unit.kelvin, 5512)
  ctx_mem.step(2)
  t1 = time.perf_counter()
  ctx_mem.step(n_steps)
  st_mem = ctx_mem.getState(getPositions=True)
  mem_ms = ((time.perf_counter() - t1) / n_steps) * 1000.0
  mem_vol_nm3 = float(st_mem.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3))

  # 3. 4.0 fs Hydrogen Mass Repartitioning (m_H = 3.0 amu) + Zero-Copy JAX/DLPack
  sys_hmr, top_hmr, pos_hmr_nm, box_hmr_nm = _build_water_box_system(
    n_waters,
    shape_kind="cubic",
    edge_nm=edge_nm,
    cutoff_nm=0.80,
    pme_grid=grid_s5,
    hmr_hydrogen_mass_amu=3.0,
    seed=5503,
  )
  sim_hmr = ProlixSimulation(
    top_hmr,
    sys_hmr,
    mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds),
    config=cfg,
  )
  sim_hmr.context.setPositions(pos_hmr_nm * unit.nanometer)
  sim_hmr.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 5513)
  sim_hmr.step(2)
  t2 = time.perf_counter()
  sim_hmr.step(n_steps)
  pos_jax_end = sim_hmr.getPositionsJax(in_nm=True)
  jax.block_until_ready(pos_jax_end)
  hmr_ms = ((time.perf_counter() - t2) / n_steps) * 1000.0
  hmr_ns_per_day = (0.004 * 86_400_000.0) / max(hmr_ms * 1000.0, 1e-9)
  hmr_bond_err_a = _max_triclinic_water_bond_err_a(
    np.asarray(pos_jax_end) * 10.0, box_hmr_nm * 10.0
  )

  # Benchmark zero-copy JAX/DLPack vs host NumPy getState
  n_io_iters = 2 if cpu_smoke else 50
  t_jc0 = time.perf_counter()
  for _ in range(n_io_iters):
    pj = sim_hmr.getPositionsJax(in_nm=True)
    vj = sim_hmr.getVelocitiesJax(in_nm_per_ps=True)
    fj = sim_hmr.getForcesJax(in_kj_per_mol_nm=True)
    sim_hmr.setPositionsJax(_DLPackTensorView(pj), in_nm=True)
    sim_hmr.setVelocitiesJax(vj, in_nm_per_ps=True)
  jax.block_until_ready(fj)
  zerocopy_us = ((time.perf_counter() - t_jc0) / n_io_iters) * 1.0e6

  pj_rt = sim_hmr.getPositionsJax(in_nm=True)
  dlpack_rt_err = float(np.max(np.abs(np.asarray(pj_rt) - np.asarray(pj))))

  t_np0 = time.perf_counter()
  for _ in range(n_io_iters):
    st_h = sim_hmr.context.getState(getPositions=True, getVelocities=True, getForces=True)
    p_np = st_h.getPositions(asNumpy=True)
    v_np = st_h.getVelocities(asNumpy=True)
    _ = st_h.getForces(asNumpy=True)
    sim_hmr.context.setPositions(p_np)
    sim_hmr.context.setVelocities(v_np)
  host_numpy_us = ((time.perf_counter() - t_np0) / n_io_iters) * 1.0e6

  integ_o = mm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
  )
  ctx_o, prov_o = _openmm_context_for_mode(
    sys_hmr, integ_o, expected_precision="mixed", cpu_smoke=cpu_smoke
  )
  ctx_o.setPositions(pos_hmr_nm * unit.nanometer)
  integ_o.step(2)
  del ctx_o, integ_o

  return {
    "n_waters": n_waters,
    "n_steps": n_steps,
    "aniso_ms_per_step": float(aniso_ms),
    "aniso_volume_nm3": aniso_vol_nm3,
    "membrane_ms_per_step": float(mem_ms),
    "membrane_volume_nm3": mem_vol_nm3,
    "hmr_dt_fs": 4.0,
    "hmr_4fs_ms_per_step": float(hmr_ms),
    "hmr_effective_ns_per_day": float(hmr_ns_per_day),
    "hmr_bond_max_err_a": float(hmr_bond_err_a),
    "zerocopy_jax_us": float(zerocopy_us),
    "host_numpy_us": float(host_numpy_us),
    "dlpack_roundtrip_max_err": dlpack_rt_err,
    "openmm_provenance": prov_o,
  }


def run_v39_complete_benchmark(
  *,
  suites: Sequence[str] = ("S1", "S2", "S3", "S4", "S5"),
  gpu_tag: str | None = None,
  quick: bool = False,
  cpu_smoke: bool = False,
) -> dict[str, Any]:
  """Run requested Pallas v3.9 benchmark suites and return structured JSON payload."""
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
    "schema_version": "pallas_v39_complete_suite_v1",
    "gpu_tag": eff_gpu_tag,
    "suites_requested": ",".join(norm_suites),
    "device_kind": str(dev0.device_kind),
    "jax_version": str(jax.__version__),
    "openmm_inventory": get_openmm_platform_inventory(),
    "cpu_smoke": bool(cpu_smoke),
    "quick_mode": bool(quick),
  }

  if "S1" in norm_suites:
    results["s1_triclinic_explicit_solvent"] = run_suite_s1_triclinic_explicit(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S2" in norm_suites:
    results["s2_ragged_cross_shape_batch"] = run_suite_s2_ragged_cross_shape_batch(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S3" in norm_suites:
    results["s3_prolate_c2_splitting"] = run_suite_s3_prolate_c2_splitting(
      cpu_smoke=cpu_smoke, quick=quick
    )
  if "S4" in norm_suites:
    results["s4_sparse_interaction_groups_alchemical"] = (
      run_suite_s4_sparse_interaction_groups_alchemical(cpu_smoke=cpu_smoke, quick=quick)
    )
  if "S5" in norm_suites:
    results["s5_barostats_hmr_and_dlpack"] = run_suite_s5_barostats_hmr_and_dlpack(
      cpu_smoke=cpu_smoke, quick=quick
    )

  results["total_wall_s"] = float(time.perf_counter() - t_wall0)
  return results


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Turnkey Pallas v3.9 Triclinic, Prolate Splitting, Alchemical & DLPack Suite."
  )
  parser.add_argument(
    "--out",
    "--output",
    dest="out",
    type=Path,
    default=Path("tmp/bench_v39_triclinic_alchem_prolate.json"),
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
  payload = run_v39_complete_benchmark(
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
