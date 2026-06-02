"""Phase 5 OpenMM Oracle — Validates OpenMM LangevinMiddleIntegrator as reference for R-step testing.

Runs TIP3P water box (895 waters, liquid density) with OpenMM's LangevinMiddleIntegrator
using SETTLE rigid-body constraints. Measures mean temperature and stability (std dev) over
the production window to validate that OpenMM stays within ±5 K of 300 K target.

This oracle serves as a trustworthy reference for prolix R-step validation (Phase 5).

Usage (standalone):
    uv run python scripts/experiments/openmm_oracle_tip3p.py --out /tmp/oracle.json --smoke

Usage (via bathos):
    uv run bth run python scripts/experiments/openmm_oracle_tip3p.py --out outputs/oracle.json --campaign ef45b8b4 -- (no extra args)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import openmm
    from openmm import unit as omm_unit
    from openmm.app import ForceField, PDBFile, PME, HBonds
    from openmm.vec3 import Vec3

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


def _build_tip3p_system(
    positions: np.ndarray,
    box_edge: float,
) -> tuple[openmm.System, openmm.Context]:
    """Build OpenMM System with TIP3P water, PME electrostatics, and SETTLE constraints.

    Args:
        positions: (N, 3) array of Cartesian coordinates in Angstroms
        box_edge: cubic box edge length in Angstroms

    Returns:
        (system, context) tuple ready for integration
    """
    # Load ForceField with TIP3P model
    ff = ForceField("tip3p.xml")

    # Build topology: TIP3P water assumes 3 atoms per molecule (O, H, H)
    n_atoms = len(positions)
    n_waters = n_atoms // 3
    if n_atoms % 3 != 0:
        raise ValueError(f"Expected n_atoms % 3 == 0, got {n_atoms}")

    # Create a minimal topology: just water residues with O, H, H atoms
    from openmm import app
    topology = app.Topology()
    chain = topology.addChain()
    for i in range(n_waters):
        residue = topology.addResidue("WAT", chain)
        topology.addAtom("O", app.element.Element.getByAtomicNumber(8), residue)
        topology.addAtom("H", app.element.Element.getByAtomicNumber(1), residue)
        topology.addAtom("H", app.element.Element.getByAtomicNumber(1), residue)

    # Create system from topology and forcefield
    system = ff.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=9.0 * omm_unit.angstroms,
        constraints=HBonds,  # SETTLE-equivalent: rigid water geometry
    )

    # Set box vectors (cubic)
    box_vectors = (
        Vec3(box_edge, 0.0, 0.0) * omm_unit.angstroms,
        Vec3(0.0, box_edge, 0.0) * omm_unit.angstroms,
        Vec3(0.0, 0.0, box_edge) * omm_unit.angstroms,
    )
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    # Create context with Langevin integrator
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * omm_unit.kelvin,
        10.0 / omm_unit.picoseconds,  # gamma = 10 ps^-1
        0.5 * omm_unit.femtoseconds,  # dt = 0.5 fs
    )

    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)

    # Set positions (in Angstroms)
    context.setPositions(positions * omm_unit.angstroms)

    return system, integrator, context


def _measure_temperature(context: openmm.Context, n_dof: int) -> float:
    """Compute T from kinetic energy using rigid-body DOF count.

    Args:
        context: OpenMM context with state
        n_dof: degrees of freedom (6*n_waters - 3 for rigid TIP3P)

    Returns:
        Temperature in Kelvin
    """
    state = context.getState(getKineticEnergy=True)
    ke_j = state.getKineticEnergy().value_in_unit(omm_unit.joules)

    # Boltzmann constant: J / K
    k_B = 1.380649e-23

    # T = 2 * KE / (k_B * DOF)
    t_k = (2.0 * ke_j) / (k_B * n_dof)
    return float(t_k)


def main() -> None:
    p = argparse.ArgumentParser(description="OpenMM oracle for TIP3P liquid-density NVT validation")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--smoke", action="store_true", help="Dry-run: no simulation, return dummy result")
    p.add_argument("--n-waters", type=int, default=895, help="Number of water molecules (default 895)")
    p.add_argument("--steps", type=int, default=3000, help="Total integration steps (default 3000)")
    p.add_argument("--burn", type=int, default=1000, help="Burn-in steps before production (default 1000)")
    p.add_argument("--dt-fs", type=float, default=0.5, help="Timestep in fs (default 0.5)")
    p.add_argument("--gamma-ps", type=float, default=10.0, help="Friction coefficient in ps^-1 (default 10.0)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for water positions (default 42)")

    args = p.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_OPENMM:
        result = {
            "mean_t": 0.0,
            "std_t": 0.0,
            "gate_pass": 0,
            "n_steps": 0,
            "error": "OpenMM not installed",
        }
        out.write_text(json.dumps(result, indent=2))
        print("Error: OpenMM not installed", file=sys.stderr)
        sys.exit(1)

    if args.smoke:
        result = {
            "mean_t": 0.0,
            "std_t": 0.0,
            "gate_pass": 0,
            "n_steps": 0,
            "smoke": True,
        }
        out.write_text(json.dumps(result, indent=2))
        print("smoke ok")
        return

    try:
        # Load equilibrated water positions
        from tests.physics.test_explicit_langevin_tip3p_parity import _equil_water_positions

        positions, box_edge = _equil_water_positions(args.n_waters, seed=args.seed)
        print(f"Loaded {args.n_waters} waters from equilibrated box (edge={box_edge} Å)")

        # Build OpenMM system and context
        system, integrator, context = _build_tip3p_system(positions, box_edge)
        n_dof = 6 * args.n_waters - 3  # rigid TIP3P: 6 DOF per water minus 3 COM

        print(f"System: {args.n_waters} waters, {n_dof} DOF, dt={args.dt_fs} fs, γ={args.gamma_ps} ps^-1")

        # Run integration and collect temperatures
        temperatures = []
        for step in range(args.steps):
            integrator.step(1)

            # Measure T every step (expensive but needed for std dev)
            if step >= args.burn:
                t_k = _measure_temperature(context, n_dof)
                temperatures.append(t_k)

            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}", flush=True)

        if not temperatures:
            result = {
                "mean_t": 0.0,
                "std_t": 0.0,
                "gate_pass": 0,
                "n_steps": args.steps,
                "error": "No production steps collected",
            }
            out.write_text(json.dumps(result, indent=2))
            print("Error: no production steps", file=sys.stderr)
            sys.exit(1)

        # Compute mean and std
        temps_array = np.array(temperatures)
        mean_t = float(np.mean(temps_array))
        std_t = float(np.std(temps_array))

        # Gate: mean_t in [295, 305] AND std_t < 10 K
        gate_pass = int((295.0 <= mean_t <= 305.0) and (std_t < 10.0))

        result = {
            "mean_t": mean_t,
            "std_t": std_t,
            "gate_pass": gate_pass,
            "n_steps": args.steps,
            "n_burn": args.burn,
            "n_waters": args.n_waters,
            "n_dof": n_dof,
        }

        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))

        if gate_pass:
            print(f"✓ Oracle PASS: T={mean_t:.1f}±{std_t:.1f} K (target 300±5 K)")
            sys.exit(0)
        else:
            print(f"✗ Oracle FAIL: T={mean_t:.1f}±{std_t:.1f} K (target 300±5 K, std<10)")
            sys.exit(1)

    except Exception as e:
        result = {
            "mean_t": 0.0,
            "std_t": 0.0,
            "gate_pass": 0,
            "n_steps": args.steps,
            "error": str(e),
        }
        out.write_text(json.dumps(result, indent=2))
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
