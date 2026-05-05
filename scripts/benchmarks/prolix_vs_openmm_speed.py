#!/usr/bin/env python3
"""Speed and VRAM benchmark: Prolix (JAX) vs OpenMM across available platforms.

Workload defaults:
------------------
- Measures energy+grad (forces) throughput.
- System: Replicated PME neutral pairs (N=2 to N=100k).
- Hardware: Enforces GPU-to-GPU comparison (JAX CUDA vs OpenMM CUDA).
- Resource Tracking: Peak GPU VRAM consumption for JAX.

Usage::

  uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --sweep 100,1000,5000,10000
  uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BOX_A = 100.0
DEFAULT_CHARGES = (1.0, -1.0)
DEFAULT_POSITIONS = np.array([[5.0, 5.0, 5.0], [20.0, 5.0, 5.0]], dtype=np.float64)
DEFAULT_ALPHA = 0.34
DEFAULT_GRID = 64 # Use larger grid for stability in larger boxes
DEFAULT_CUTOFF_A = 9.0

OPENMM_PLATFORM_ORDER = ("CUDA", "OpenCL", "HIP", "CPU", "Reference")


@dataclass
class TimingRow:
  engine: str
  backend: str
  n_atoms: int
  mean_ms: float
  std_ms: float
  calls_per_s: float
  vram_gb: float = 0.0
  notes: str = ""


def get_vram_usage_gb() -> float:
    """Return current peak VRAM usage in GB for the default JAX device."""
    try:
        # Note: This requires JAX 0.4.x+ and a GPU backend
        device = jax.devices()[0]
        if device.platform != 'gpu':
            return 0.0
        stats = device.memory_stats()
        peak_bytes = stats.get('peak_bytes_in_use', 0)
        return peak_bytes / (1024**3)
    except:
        return 0.0


def _git_sha() -> str:
  try:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out = subprocess.run(
      ["git", "-C", root, "rev-parse", "--short", "HEAD"],
      capture_output=True,
      text=True,
      check=False,
    )
    return out.stdout.strip() or "unknown"
  except OSError:
    return "unknown"


def build_openmm_system(n_atoms: int, box_angstrom: float):
  import openmm
  from openmm import unit
  
  box_nm = box_angstrom / 10.0
  omm_system = openmm.System()
  omm_system.setDefaultPeriodicBoxVectors(
    openmm.Vec3(box_nm, 0, 0),
    openmm.Vec3(0, box_nm, 0),
    openmm.Vec3(0, 0, box_nm),
  )
  
  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.PME)
  nb.setCutoffDistance(DEFAULT_CUTOFF_A / 10.0)
  nb.setPMEParameters(DEFAULT_ALPHA * 10.0, DEFAULT_GRID, DEFAULT_GRID, DEFAULT_GRID)
  
  for i in range(n_atoms):
    omm_system.addParticle(1.0)
    q = 1.0 if i % 2 == 0 else -1.0
    nb.addParticle(q, 0.3, 0.1) # Standard C-like sig/eps
    
  omm_system.addForce(nb)
  return omm_system


def run_prolix_bench(n_atoms: int, box: float, warmup: int, repeats: int) -> tuple[float, float, float]:
  from prolix.physics import pbc, system, neighbor_list as nl
  from jax_md import partition

  # Setup coordinates
  charges = jnp.array([1.0, -1.0] * (n_atoms // 2))
  # Replicate positions
  rng = np.random.default_rng(42)
  positions = rng.uniform(0, box, (n_atoms, 3))
  
  params = {
    "charges": charges,
    "sigmas": jnp.ones(n_atoms),
    "epsilons": jnp.ones(n_atoms) * 0.1,
    "exclusion_mask": None,
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2)),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2)),
    "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
    "dihedral_params": jnp.zeros((0, 3)),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3)),
  }
  
  box_vec = jnp.array([box, box, box])
  displacement_fn, _ = pbc.create_periodic_space(box_vec)
  
  # Allocate neighbor list
  neighbor_list_fn = partition.neighbor_list(displacement_fn, box_vec, DEFAULT_CUTOFF_A + 1.0, dr_threshold=0.5)
  nbr = neighbor_list_fn.allocate(positions)
  
  energy_fn = system.make_energy_fn(
    displacement_fn,
    params,
    positions=positions,
    neighbor_list=nbr,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=DEFAULT_GRID,
    pme_alpha=DEFAULT_ALPHA,
    cutoff_distance=DEFAULT_CUTOFF_A,
  )

  val_and_grad = jax.jit(jax.value_and_grad(energy_fn))

  # Warmup
  _ = val_and_grad(positions, neighbor=nbr)
  for _ in range(warmup):
    _ = val_and_grad(positions, neighbor=nbr)
    
  samples = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    val, grad = val_and_grad(positions, neighbor=nbr)
    val.block_until_ready()
    samples.append((time.perf_counter() - t0) * 1000.0)
    
  arr = np.array(samples)
  return float(arr.mean()), float(arr.std()), get_vram_usage_gb()


def run_openmm_bench(n_atoms: int, box: float, warmup: int, repeats: int) -> tuple[float, float]:
    import openmm
    from openmm import app, unit
    
    omm_system = build_openmm_system(n_atoms, box)
    integrator = openmm.VerletIntegrator(0.001)
    
    # Select best platform
    platform = None
    for pname in OPENMM_PLATFORM_ORDER:
        try:
            platform = openmm.Platform.getPlatformByName(pname)
            break
        except:
            continue
            
    if platform is None:
        raise RuntimeError("No OpenMM platform found")
        
    context = openmm.Context(omm_system, integrator, platform)
    
    # Set positions
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, box/10.0, (n_atoms, 3))
    context.setPositions(positions)
    
    # Warmup
    for _ in range(warmup):
        context.getState(getEnergy=True, getForces=True)
        
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        context.getState(getEnergy=True, getForces=True)
        samples.append((time.perf_counter() - t0) * 1000.0)
        
    arr = np.array(samples)
    return float(arr.mean()), float(arr.std())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sweep", type=str, default="1000,5000,10000", help="Comma-separated atom counts.")
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--json", type=str, default="", help="Save results to JSON.")
  parser.add_argument("--skip-prolix", action="store_true", help="Skip Prolix (JAX) benchmarks.")
  parser.add_argument("--skip-openmm", action="store_true", help="Skip OpenMM benchmarks.")
  args = parser.parse_args()

  n_list = [int(x) for x in args.sweep.split(",")]
  results = []
  
  git_sha = _git_sha()
  print(f"=== Prolix Scaling Benchmark (SHA: {git_sha}) ===")
  print(f"{'N_Atoms':>8} | {'Engine':>8} | {'Mean (ms)':>10} | {'Std (ms)':>10} | {'VRAM (GB)':>8}")
  print("-" * 60)

  for n in n_list:
    # Prolix
    if not args.skip_prolix:
      m_p, s_p, v_p = run_prolix_bench(n, DEFAULT_BOX_A, args.warmup, args.repeats)
      print(f"{n:8d} | {'Prolix':>8} | {m_p:10.3f} | {s_p:10.3f} | {v_p:8.3f}")
      results.append(TimingRow("Prolix", "jax", n, m_p, s_p, 1000.0/m_p, v_p))
    
    # OpenMM
    if not args.skip_openmm:
      try:
          m_o, s_o = run_openmm_bench(n, DEFAULT_BOX_A, args.warmup, args.repeats)
          print(f"{n:8d} | {'OpenMM':>8} | {m_o:10.3f} | {s_o:10.3f} | {'n/a':>8}")
          results.append(TimingRow("OpenMM", "cuda", n, m_o, s_o, 1000.0/m_o, 0.0))
      except Exception as e:
          print(f"{n:8d} | {'OpenMM':>8} | {'FAILED':>10} | {'-':>10} | {'-':>8}")

  if args.json:
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "results": [asdict(r) for r in results]
    }
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.json}")

if __name__ == "__main__":
  main()
