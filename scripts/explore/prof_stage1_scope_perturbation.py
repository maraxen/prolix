#!/usr/bin/env python3
"""P5 -- Perturbation check: do P2's new named_scope wraps change fused hot-path timing?

See .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section P5. The CPU
gate on P2: necessary but not sufficient (fusion decisions are backend- and shape-dependent;
P7 carries the Stage-2/H200 replication).

Process model (subprocess-per-arm, since an import-time patch cannot be flipped in-process):
this script is both the driver AND its own child. The driver alternately spawns
``--child --arm scopes_on`` / ``--arm scopes_off`` invocations of itself, A/B/A/B. Each child
patches ``jax.named_scope`` to a null contextmanager (scopes_off) or leaves it alone
(scopes_on) BEFORE importing any prolix module, builds a deterministic N=64 synthetic system
from ``--system-seed`` alone, runs 3 untimed warm-ups (absorbing JIT compile) then 20 timed
repeats of the target function, reports the median wall time + writes ``forces.npy`` beside
its ``ProbeRecord``. The driver compares both arms' forces (``numpy.allclose``, NOT bitwise --
a no-op scope cannot change semantics, but it can change XLA fusion, and a changed fusion
changes reduction/FMA-contraction order), computes paired relative deltas
``d_i = (t_on_i - t_off_i) / t_off_i``, and gates on the median + a percentile-bootstrap CI.

Two independent checks, both must pass: (1) --target whole_step, a synthetic construct that
sums dense-path forces (single_padded_energy's group iii: dense_bonded_*/dense_lj_direct/
dense_coulomb_direct/dense_pme_reciprocal) and flash-path forces (flash_explicit_forces's
groups i+ii: flash_bonded/flash_grad and flash_nonbonded_tiles/flash_lj_only) so a
perturbation in ANY P2 scope group is detectable by one measurement -- not a physically
meaningful quantity, purely a harness construct matching the spec's own localisation
procedure, which bisects across exactly these three groups if this check fails.
(2) --target tile_function, chunked_explicit_nonbonded_energy timed alone, because a
perturbation confined to the tile reduction can hide under a whole-step's 5% margin.

If either check DETECTS a perturbation (|median(d_i)| > 0.05), the spec's localisation
procedure is a MANUAL git-branch bisection (revert one of the three scope groups' wrapping in
a scratch branch, re-run the whole-step check at the pilot-derived n_pairs, repeat per group)
-- not automated here, since the spec explicitly describes it as a source-level revert, not a
runtime toggle.

Usage::

    uv run python scripts/explore/prof_stage1_scope_perturbation.py
    uv run python scripts/explore/prof_stage1_scope_perturbation.py --target tile_function
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys as _sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

N_ATOMS = 64
N_WARMUP = 3
N_INNER = 20
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260817
TOLERANCE_MARGIN = 0.05
PILOT_N_PAIRS = 10
N_PAIRS_MIN = 21
N_PAIRS_MAX = 200
N_DISCARD = 2
BOX_SIZE_VALUE = 50.0
PME_ALPHA = 0.3


def _synthetic_system(seed: int):
    """A deterministic, topology-free N=64 PhysicsSystem from ``seed`` alone.

    Matches P3's (scripts/explore/prof_stage0_cost_analysis.py) construction
    exactly -- both arms of a pair are given the SAME seed, so they are
    bit-identical inputs; only the scopes_on/scopes_off patch differs.
    """
    import jax
    import jax.numpy as jnp

    from prolix.typing import PhysicsSystem

    pos_key = jax.random.key(seed)
    positions = jax.random.normal(pos_key, (N_ATOMS, 3), dtype=jnp.float32) * 10.0
    ones_b = jnp.ones(N_ATOMS, dtype=bool)
    zeros_b = jnp.zeros(N_ATOMS, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    empty4 = jnp.zeros((0, 4), dtype=jnp.int32)
    empty_p2 = jnp.zeros((0, 2), dtype=jnp.float32)
    empty_dih_p = jnp.zeros((0, 1, 3), dtype=jnp.float32)
    empty_m = jnp.zeros(0, dtype=bool)
    excl_indices = jnp.zeros((N_ATOMS, 1), dtype=jnp.int32)
    excl_scales = jnp.ones((N_ATOMS, 1), dtype=jnp.float32)

    return PhysicsSystem(
        positions=positions,
        charges=jnp.zeros(N_ATOMS, dtype=jnp.float32),
        sigmas=jnp.full((N_ATOMS,), 3.15, dtype=jnp.float32),
        epsilons=jnp.full((N_ATOMS,), 0.1, dtype=jnp.float32),
        radii=jnp.ones(N_ATOMS, dtype=jnp.float32),
        scaled_radii=jnp.ones(N_ATOMS, dtype=jnp.float32),
        masses=jnp.full((N_ATOMS,), 12.0, dtype=jnp.float32),
        element_ids=jnp.zeros(N_ATOMS, dtype=jnp.int32),
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
        box_size=jnp.array([BOX_SIZE_VALUE, BOX_SIZE_VALUE, BOX_SIZE_VALUE], dtype=jnp.float32),
        pme_alpha=PME_ALPHA,
    )


def _whole_step_forces(sys):
    """Touches every P2 scope group in one call: dense (group iii) + flash
    (groups i+ii) forces, summed. Not a physically meaningful quantity --
    see this module's docstring for why."""
    import equinox as eqx

    from prolix.batched_energy import single_padded_energy
    from prolix.physics import pbc
    from prolix.physics.flash_explicit import flash_explicit_forces

    disp_fn, _ = pbc.create_periodic_space(sys.box_size)

    @eqx.filter_grad
    def _dense_grad(s):
        return single_padded_energy(s, disp_fn, implicit_solvent=False)

    dense_forces = -_dense_grad(sys).positions
    flash_forces = flash_explicit_forces(sys)
    return dense_forces + flash_forces


def _tile_function_forces(sys):
    """chunked_explicit_nonbonded_energy (group ii's tile reduction) alone."""
    import jax

    from prolix.physics.flash_explicit import chunked_explicit_nonbonded_energy

    def _energy(positions):
        return chunked_explicit_nonbonded_energy(
            positions,
            sys.charges,
            sys.sigmas,
            sys.epsilons,
            sys.atom_mask,
            sys,
            pme_alpha=sys.pme_alpha,
        )

    return -jax.grad(_energy)(sys.positions)


_TARGET_FNS = {"whole_step": _whole_step_forces, "tile_function": _tile_function_forces}


class _NullScope(contextlib.ContextDecorator):
    """A no-op standing in for jax.named_scope, usable as BOTH a context
    manager (``with jax.named_scope("x"):``, prolix's own usage) and a
    decorator (``@named_scope("x")``, confirmed needed: equinox's own
    ``eqx.nn`` module decorates methods with named_scope at import time --
    ``contextlib.nullcontext`` alone is not callable and breaks that import).
    """

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False


def _run_child(arm: str, system_seed: int, target: str, out_path: Path) -> None:
    if arm == "scopes_off":
        import jax

        # Deliberate monkeypatch, not a bug -- this IS the scopes_off arm's
        # mechanism (spec P5: "patching jax.named_scope to a null
        # contextmanager"). ty flags the type mismatch against the real
        # named_scope signature, which is expected for a monkeypatch.
        jax.named_scope = lambda *_a, **_k: _NullScope()  # ty: ignore[invalid-assignment]
    elif arm != "scopes_on":
        raise ValueError(f"unknown arm {arm!r}")

    import jax
    import numpy as np

    from scripts.profiling.record import ProbeRecord

    sys_obj = _synthetic_system(system_seed)
    # jax.jit is essential, not incidental: the whole point of this gate is
    # whether a no-op named_scope changes FUSED hot-path timing -- fusion is
    # an XLA compile-time decision that only exists under jit. Un-jitted,
    # every call (including the 3 "warm-up" ones) pays full eager per-op
    # dispatch cost and the warm-ups absorb nothing (confirmed: median
    # ~0.86s per call on this tiny N=64 system before this fix, ~1000x
    # slower than a compiled call should be).
    fn = jax.jit(_TARGET_FNS[target])

    for _ in range(N_WARMUP):
        result = fn(sys_obj)
        result.block_until_ready()

    times: list[float] = []
    forces = None
    for _ in range(N_INNER):
        t0 = time.perf_counter()
        result = fn(sys_obj)
        result.block_until_ready()
        times.append(time.perf_counter() - t0)
        forces = result

    median_time = float(np.median(times))
    forces_np = np.asarray(forces, dtype=np.float64)

    record = ProbeRecord(
        probe_id=f"stage1_pert_{arm}_{target}_seed{system_seed}",
        stage=1,
        n_atoms=N_ATOMS,
        platform="cpu",
        metrics={"median_step_seconds": median_time},
        config={"arm": arm, "target": target, "system_seed": str(system_seed)},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record.write(out_path)
    np.save(str(out_path) + ".forces.npy", forces_np)

    print(
        json.dumps({"median_seconds": median_time, "x64_enabled": record.x64_enabled}),
        flush=True,
    )


def _spawn_child(
    arm: str, system_seed: int, target: str, out_path: Path
) -> dict:
    cmd = [
        _sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arm",
        arm,
        "--system-seed",
        str(system_seed),
        "--target",
        target,
        "--out",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def _run_pair(system_seed: int, target: str, out_dir: Path, pair_idx: int) -> dict:
    """Run one A/B pair (scopes_on then scopes_off) and compare forces."""
    import numpy as np

    on_path = out_dir / f"pair{pair_idx}_on.json"
    off_path = out_dir / f"pair{pair_idx}_off.json"
    on_summary = _spawn_child("scopes_on", system_seed, target, on_path)
    off_summary = _spawn_child("scopes_off", system_seed, target, off_path)

    assert on_summary["x64_enabled"] == off_summary["x64_enabled"], (
        "cross-precision A/B pair -- a delta between arms measured under "
        "different x64_enabled settings is void regardless of the value"
    )
    x64_enabled = on_summary["x64_enabled"]
    rtol, atol = (1e-12, 1e-14) if x64_enabled else (1e-5, 1e-6)

    f_on = np.load(str(on_path) + ".forces.npy")
    f_off = np.load(str(off_path) + ".forces.npy")
    close = bool(np.allclose(f_on, f_off, rtol=rtol, atol=atol))
    max_abs_delta = float(np.max(np.abs(f_on - f_off)))

    t_on = on_summary["median_seconds"]
    t_off = off_summary["median_seconds"]
    delta = (t_on - t_off) / t_off

    return {
        "delta": delta,
        "allclose": close,
        "max_abs_force_delta": max_abs_delta,
        "x64_enabled": x64_enabled,
    }


def _bootstrap_median_ci(
    deltas, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED, ci: float = 0.90
):
    import numpy as np

    rng = np.random.default_rng(seed)
    deltas_arr = np.asarray(deltas)
    n = len(deltas_arr)
    resamples = rng.choice(deltas_arr, size=(B, n), replace=True)
    boot_medians = np.median(resamples, axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_medians, alpha))
    hi = float(np.quantile(boot_medians, 1 - alpha))
    return lo, hi


def _run_gate(
    target: str, out_dir: Path, base_seed: int, n_pairs_override: int | None = None
) -> dict:
    import math

    import numpy as np

    # Per-target subdirectory -- required, not cosmetic: two targets (or two
    # re-runs) sharing one out_dir would both write pair{i}_on.json/etc at
    # the SAME path, racing if run concurrently (confirmed: this bug was
    # caught when whole_step and tile_function were launched as separate
    # background processes against the same out_dir and clobbered each
    # other's pair files).
    target_dir = out_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== target={target}: pilot ({PILOT_N_PAIRS} A/A pairs) ===")
    pilot_deltas = []
    for i in range(PILOT_N_PAIRS):
        seed = base_seed + i
        # A/A: both arms scopes_off -- pure null/noise-floor measurement.
        on_path = target_dir / f"pilot{i}_a.json"
        off_path = target_dir / f"pilot{i}_b.json"
        a = _spawn_child("scopes_off", seed, target, on_path)
        b = _spawn_child("scopes_off", seed, target, off_path)
        pilot_deltas.append((a["median_seconds"] - b["median_seconds"]) / b["median_seconds"])

    pilot_deltas = np.asarray(pilot_deltas)
    q75, q25 = np.percentile(pilot_deltas, [75, 25])
    s = float(q75 - q25)
    raw_n_pairs = math.ceil((3.3 * s / TOLERANCE_MARGIN) ** 2) if s > 0 else N_PAIRS_MIN
    n_pairs = int(min(max(raw_n_pairs, N_PAIRS_MIN), N_PAIRS_MAX))
    if n_pairs_override is not None:
        # Per spec: an INSUFFICIENT_POWER verdict means "increase n_pairs
        # and re-run; DO NOT revert" -- this is that escalation path, not a
        # way to dodge a DETECTED_PERTURBATION verdict (which is not
        # power-sensitive: it fires on the median itself, not the CI width).
        print(f"n_pairs override: {n_pairs} -> {n_pairs_override}")
        n_pairs = int(max(n_pairs_override, N_PAIRS_MIN))
    unresolved_on_cpu = raw_n_pairs > N_PAIRS_MAX
    print(f"pilot IQR s={s:.6g}, raw_n_pairs={raw_n_pairs}, clipped n_pairs={n_pairs}")

    print(f"=== target={target}: {n_pairs} interleaved A/B pairs ===")
    pair_results = []
    for i in range(n_pairs):
        seed = base_seed + 1000 + i
        pair_results.append(_run_pair(seed, target, target_dir, i))

    if not all(r["allclose"] for r in pair_results):
        bad = [i for i, r in enumerate(pair_results) if not r["allclose"]]
        raise AssertionError(
            f"target={target}: allclose FAILED on pair(s) {bad} -- the scopes_off "
            "patch disabled or reordered real work; the timing comparison is void "
            "until this is fixed"
        )

    kept = pair_results[N_DISCARD:]
    deltas = np.asarray([r["delta"] for r in kept])
    median_delta = float(np.median(deltas))
    ci_lo, ci_hi = _bootstrap_median_ci(deltas)
    max_abs_force_delta = max(r["max_abs_force_delta"] for r in pair_results)

    within_median = abs(median_delta) <= TOLERANCE_MARGIN
    within_ci = (-TOLERANCE_MARGIN <= ci_lo) and (ci_hi <= TOLERANCE_MARGIN)

    if unresolved_on_cpu:
        verdict = "UNRESOLVED_ON_CPU"
    elif within_median and within_ci:
        verdict = "PASS"
    elif within_median and not within_ci:
        verdict = "INSUFFICIENT_POWER"
    else:
        verdict = "DETECTED_PERTURBATION"

    return {
        "target": target,
        "verdict": verdict,
        "median_delta": median_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_pairs": n_pairs,
        "pilot_s": s,
        "pilot_raw_n_pairs": raw_n_pairs,
        "max_abs_force_delta": max_abs_force_delta,
        "x64_enabled": pair_results[0]["x64_enabled"],
    }


def _emit_summary_record(result: dict, out_dir: Path) -> None:
    from scripts.profiling.record import ProbeRecord

    record = ProbeRecord(
        probe_id=f"stage1_pert_summary_{result['target']}",
        stage=1,
        n_atoms=N_ATOMS,
        platform="cpu",
        metrics={
            "median_relative_delta": result["median_delta"],
            "ci_lo": result["ci_lo"],
            "ci_hi": result["ci_hi"],
            "n_pairs": result["n_pairs"],
            "n_inner": N_INNER,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_b": BOOTSTRAP_B,
            "pilot_s": result["pilot_s"],
            "pilot_raw_n_pairs": result["pilot_raw_n_pairs"],
            "ab_max_abs_force_delta": result["max_abs_force_delta"],
        },
        config={
            "target": result["target"],
            "method": "percentile bootstrap",
            "verdict": result["verdict"],
        },
    )
    out_path = out_dir / f"{record.probe_id}.json"
    record.write(out_path)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=["scopes_on", "scopes_off"])
    parser.add_argument("--system-seed", type=int)
    parser.add_argument("--target", choices=list(_TARGET_FNS), default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--out-dir", default=str(_ROOT / "outputs" / "profiling" / "stage1_perturbation")
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help=(
            "Override the pilot-derived n_pairs. Use to re-run after an "
            "INSUFFICIENT_POWER verdict with a larger n_pairs (spec: "
            "increase n_pairs and re-run, do NOT revert). Does not change "
            "the DETECTED_PERTURBATION criterion, which is not power-"
            "sensitive."
        ),
    )
    args = parser.parse_args()

    if args.child:
        assert args.arm and args.system_seed is not None and args.target and args.out
        _run_child(args.arm, args.system_seed, args.target, args.out)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.target] if args.target else list(_TARGET_FNS)

    results = []
    for target in targets:
        result = _run_gate(target, out_dir, args.base_seed, args.n_pairs)
        _emit_summary_record(result, out_dir / target)
        results.append(result)
        print(
            f"target={target} verdict={result['verdict']} "
            f"median_delta={result['median_delta']:.4f} "
            f"ci=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}] "
            f"n_pairs={result['n_pairs']} "
            f"ab_max_abs_force_delta={result['max_abs_force_delta']:.3e}"
        )

    overall_pass = all(r["verdict"] == "PASS" for r in results)
    print(f"\nOVERALL: {'PASS' if overall_pass else 'NOT PASS'} (both checks must PASS)")


if __name__ == "__main__":
    main()
