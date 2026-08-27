#!/usr/bin/env python3
"""Stage 0 probe: cost_analysis() dense vs flash, no execution.

See .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section P3.
Structural claims only -- lowers and compiles both force paths
(``prolix.batched_energy.single_padded_energy`` + grad, and
``prolix.physics.flash_explicit.flash_explicit_forces``) for tiny synthetic
systems (N = 4, 16, 64) via ``jax.jit(fn).lower(*args).compile().cost_analysis()``,
and emits ``ProbeRecord(stage=0, platform="cpu", ...)`` records. Deliberately
never executes the compiled computation -- this is the free/no-compute step
(scripts/explore/, not scripts/experiments/ -- ad-hoc exploration, no bathos
sidecar required per the project's bathos hard rules).

Usage::

    uv run python scripts/explore/prof_stage0_cost_analysis.py
    uv run python scripts/explore/prof_stage0_cost_analysis.py --out-dir /tmp/stage0
"""

from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from scripts.profiling.claims import (  # noqa: E402
    ClaimClass,
    ClaimValidityError,
    assert_claim_supported,
)
from scripts.profiling.record import ProbeRecord  # noqa: E402

N_SIZES = (4, 16, 64)
BOX_SIZE = jnp.array([50.0, 50.0, 50.0], dtype=jnp.float32)
PME_ALPHA = 0.3


def _synthetic_system(n_atoms: int, seed: int = 0):
    """A minimal, topology-free ``PhysicsSystem`` at a literal N atom count.

    Deliberately bypasses ``MolecularBundle``/``ATOM_BUCKETS`` bucketing
    (whose smallest bucket is 64) -- this probe needs genuinely different
    N=4/16/64 shapes to compare ``cost_analysis()`` scaling structurally,
    which a bucket-floored system could never show for N<64. Zero bonds/
    angles/dihedrals/impropers keeps every host-side topology-processing
    branch (e.g. ``topology.find_bonded_exclusions``) inactive -- those
    branches are gated on ``sys.bonds.shape[0] > 0``, a static, trace-time-
    known check, so an empty-shaped array here never trips a
    TracerArrayConversionError even when this whole ``PhysicsSystem`` is
    passed directly as a ``jax.jit`` argument (as opposed to closed over,
    the pattern P2's HLO-inventory test needed for a *real*-topology
    fixture).
    """
    from prolix.typing import PhysicsSystem

    pos_key = jax.random.key(seed)
    positions = jax.random.normal(pos_key, (n_atoms, 3), dtype=jnp.float32) * 10.0
    ones_b = jnp.ones(n_atoms, dtype=bool)
    zeros_b = jnp.zeros(n_atoms, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    empty4 = jnp.zeros((0, 4), dtype=jnp.int32)
    empty_p2 = jnp.zeros((0, 2), dtype=jnp.float32)
    empty_dih_p = jnp.zeros((0, 1, 3), dtype=jnp.float32)
    empty_m = jnp.zeros(0, dtype=bool)
    # A single, no-op exclusion column (index 0, scale 1.0 == "not
    # excluded") -- both flash_explicit.py's tile kernel and
    # batched_energy.py's dense path require excl_indices/excl_scales_* to
    # be real (non-None) arrays, even with zero real exclusions.
    excl_indices = jnp.zeros((n_atoms, 1), dtype=jnp.int32)
    excl_scales = jnp.ones((n_atoms, 1), dtype=jnp.float32)

    return PhysicsSystem(
        positions=positions,
        charges=jnp.zeros(n_atoms, dtype=jnp.float32),
        sigmas=jnp.full((n_atoms,), 3.15, dtype=jnp.float32),
        epsilons=jnp.full((n_atoms,), 0.1, dtype=jnp.float32),
        radii=jnp.ones(n_atoms, dtype=jnp.float32),
        scaled_radii=jnp.ones(n_atoms, dtype=jnp.float32),
        masses=jnp.full((n_atoms,), 12.0, dtype=jnp.float32),
        element_ids=jnp.zeros(n_atoms, dtype=jnp.int32),
        atom_mask=ones_b,
        is_hydrogen=zeros_b,
        is_backbone=zeros_b,
        is_heavy=ones_b,
        protein_atom_mask=zeros_b,
        water_atom_mask=zeros_b,
        bonds=empty2,
        bond_params=empty_p2,
        bond_mask=empty_m,
        angles=empty3,
        angle_params=empty_p2,
        angle_mask=empty_m,
        dihedrals=empty4,
        dihedral_params=empty_dih_p,
        dihedral_mask=empty_m,
        impropers=empty4,
        improper_params=empty_dih_p,
        improper_mask=empty_m,
        urey_bradley_bonds=empty2,
        urey_bradley_params=empty_p2,
        urey_bradley_mask=empty_m,
        excl_indices=excl_indices,
        excl_scales_vdw=excl_scales,
        excl_scales_elec=excl_scales,
        box_size=BOX_SIZE,
        pme_alpha=PME_ALPHA,
    )


def _dense_forces_fn(sys):
    """``single_padded_energy`` + grad -- the dense O(N^2) force path."""
    from prolix.batched_energy import single_padded_energy
    from prolix.physics import pbc

    disp_fn, _ = pbc.create_periodic_space(sys.box_size)

    @eqx.filter_grad
    def _grad(s):
        return single_padded_energy(s, disp_fn, implicit_solvent=False)

    return _grad(sys)


def _flash_forces_fn(sys):
    """FlashMD tiled force path."""
    from prolix.physics.flash_explicit import flash_explicit_forces

    return flash_explicit_forces(sys)


def _cost_metrics(cost) -> dict[str, float]:
    """Normalize ``XlaExecutable.cost_analysis()``'s return shape into ``metrics``.

    JAX returns a dict, or (some versions) a single-element list of dicts --
    normalise the list form to its sole element, raise if it has more.
    Stamp ``flops``, ``bytes_accessed``, and ``optimal_seconds`` under
    exactly those names (mapping JAX's own ``"bytes accessed"`` spelling),
    omitting any the installed JAX does not provide. Any further keys are
    stamped verbatim under their JAX-given names.
    """
    if isinstance(cost, list):
        if len(cost) != 1:
            raise ValueError(
                f"cost_analysis() returned a list of {len(cost)} elements, "
                "expected exactly 1"
            )
        cost = cost[0]

    metrics: dict[str, float] = {}
    if "flops" in cost:
        metrics["flops"] = float(cost["flops"])
    if "bytes accessed" in cost:
        metrics["bytes_accessed"] = float(cost["bytes accessed"])
    if "optimal_seconds" in cost:
        metrics["optimal_seconds"] = float(cost["optimal_seconds"])

    fixed_keys = {"flops", "bytes accessed", "optimal_seconds"}
    for key, value in cost.items():
        if key in fixed_keys:
            continue
        metrics[key] = float(value)
    return metrics


def _probe_one(n_atoms: int, force_path: str) -> ProbeRecord:
    sys_obj = _synthetic_system(n_atoms)
    fn = _dense_forces_fn if force_path == "dense" else _flash_forces_fn
    compiled = jax.jit(fn).lower(sys_obj).compile()
    cost = compiled.cost_analysis()
    metrics = _cost_metrics(cost)
    return ProbeRecord(
        probe_id=f"stage0_{force_path}_N{n_atoms}",
        stage=0,
        n_atoms=n_atoms,
        platform="cpu",
        metrics=metrics,
        config={"force_path": force_path},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "outputs" / "profiling" / "stage0"),
        help="Directory to write ProbeRecord JSON files into.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    records: list[ProbeRecord] = []
    observed_cost_keys: set[str] = set()
    for n_atoms in N_SIZES:
        for force_path in ("dense", "flash"):
            record = _probe_one(n_atoms, force_path)
            records.append(record)
            observed_cost_keys.update(record.metrics.keys())
            out_path = out_dir / f"{record.probe_id}.json"
            record.write(out_path)
            print(f"wrote {out_path} metrics={sorted(record.metrics.keys())}")

    print(f"emitted {len(records)} Stage-0 records to {out_dir}")
    print(f"cost_analysis() keys observed across this JAX install: {sorted(observed_cost_keys)}")

    # Gate smoke assertion: TERM_RANKING must raise on Stage-0 records, and
    # for the right reason -- Stage-0 records carry neither
    # total_step_seconds nor scopes, so the fail-closed metric/scopes
    # selector in select_sources must fire before the stage>=2 floor in
    # assert_claim_supported is ever reached.
    try:
        assert_claim_supported(records, ClaimClass.TERM_RANKING)
    except ClaimValidityError as exc:
        message = str(exc)
        assert "stage>=2" not in message, (
            "expected the metric/scopes filter to fire first, but the stage "
            f"floor message fired instead: {message}"
        )
        assert "metrics" in message or "scopes" in message, (
            f"expected select_sources' metric/scopes filter message, got: {message}"
        )
        print(f"smoke check OK: TERM_RANKING raised for the right reason: {message}")
    else:
        raise AssertionError(
            "assert_claim_supported(records, ClaimClass.TERM_RANKING) was "
            "expected to raise on Stage-0 records, but did not"
        )


if __name__ == "__main__":
    main()
