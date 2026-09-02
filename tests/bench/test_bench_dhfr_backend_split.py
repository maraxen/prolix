"""Guards for bench_dhfr_parity's per-backend split and its combine step.

Why this split exists
---------------------
The DHFR ns/day comparison ran prolix and OpenMM in one process, and its OpenMM
leg was silently falling back to CPU -- which flatters prolix, so every ratio ever
reported from it is a ceiling on our standing rather than a floor (#295).

The two halves cannot share an environment: the CUDA-capable OpenMM lives in a
conda env with no jax, no CUDA packages and no prolix, while the workspace venv
that has jax+prolix carries the PyPI ``openmm`` wheel, which ships no CUDA plugin.
So each leg must run under its own interpreter and the results are merged
afterwards -- the same shape ``scripts/slurm/b1_init_exec.slurm`` already uses.

These tests are pure dict/JSON logic: no JAX, no GPU, no prolix import. They are
safe in CI and locally, unlike the benchmark itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXP = _ROOT / "scripts" / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import bench_dhfr_parity as B  # noqa: E402


def _prolix_leg(**over):
    base = {
        "backend": "prolix",
        "smoke": False,
        "prolix_ns_per_day": 0.779,
        "per_step_corrected_s": 0.1109,
        "compile_fixed_s": 7.8,
        "r_squared": 0.99999,
        "n_atoms": 23558,
        "n_steps_list": [50, 200, 1000, 5000],
        "raw_results": [{"n_steps": 50, "t_steady_state": 5.4}],
        "shim_mode": "autograd",
        "seed": 0,
        "dt_fs": 1.0,
        "gpu_tag": "h200",
        # an --backend prolix run emits explicit None for the other leg's fields
        "openmm_ns_per_day": None,
        "openmm_wallclock_s": None,
        "openmm_platform": None,
        "openmm_n_production_steps": None,
    }
    return {**base, **over}


def _openmm_leg(**over):
    base = {
        "backend": "openmm",
        "smoke": False,
        "openmm_ns_per_day": 12.5,
        "openmm_wallclock_s": 34.5,
        "openmm_platform": "CUDA",
        "openmm_n_production_steps": 5000,
        "gpu_tag": "h200",
        # ... and symmetrically, None for the prolix fields
        "prolix_ns_per_day": None,
        "per_step_corrected_s": None,
        "compile_fixed_s": None,
        "r_squared": None,
        "n_atoms": None,
        "n_steps_list": None,
        "raw_results": None,
    }
    return {**base, **over}


def _combine(tmp_path, prolix, openmm):
    p = tmp_path / "prolix.json"
    o = tmp_path / "openmm.json"
    p.write_text(json.dumps(prolix))
    o.write_text(json.dumps(openmm))
    return B._combine_results(p, o)


def test_module_imports_without_initialising_jax():
    """The property the whole split exists for.

    The OpenMM leg runs in an env with no jax installed at all, so importing this
    module must not import jax. If this regresses, ``--backend openmm`` dies at
    import time in the CUDA env and the real baseline becomes unobtainable again.
    """
    assert B.jax is None, (
        f"importing bench_dhfr_parity initialised JAX ({B.jax!r}); the lazy "
        "_init_jax() seam is broken and --backend openmm can no longer run in "
        "the jax-free CUDA OpenMM environment"
    )


def test_combine_keeps_both_legs_numbers(tmp_path):
    """A naive {**prolix, **openmm} merge silently destroys the prolix side.

    Each leg emits explicit None placeholders for the other leg's fields, so splat
    order decides whether real numbers or Nones survive. This asserts the merge
    takes each field from the leg that produced it -- otherwise the output reports
    a valid ratio next to prolix_ns_per_day: null, and the bathos gate reads those
    fields.
    """
    m = _combine(tmp_path, _prolix_leg(), _openmm_leg())

    assert m["prolix_ns_per_day"] == 0.779
    assert m["n_atoms"] == 23558
    assert m["r_squared"] == 0.99999
    assert m["raw_results"] == [{"n_steps": 50, "t_steady_state": 5.4}]
    assert m["openmm_ns_per_day"] == 12.5
    assert m["openmm_platform"] == "CUDA"
    assert m["ratio"] == pytest.approx(0.779 / 12.5)
    assert m["backend"] == "combined"


def test_combine_keeps_both_gpu_tags_and_drops_the_ambiguous_one(tmp_path):
    """Two runs, two tags -- a singular gpu_tag would be read as authoritative."""
    m = _combine(tmp_path, _prolix_leg(gpu_tag="h200"), _openmm_leg(gpu_tag="a100"))
    assert m["prolix_gpu_tag"] == "h200"
    assert m["openmm_gpu_tag"] == "a100"
    assert "gpu_tag" not in m


def test_combine_refuses_a_cpu_openmm_baseline(tmp_path):
    """The bug this whole sprint exists to close: CPU silently as the denominator."""
    with pytest.raises(RuntimeError, match="not 'CUDA'"):
        _combine(tmp_path, _prolix_leg(), _openmm_leg(openmm_platform="CPU"))


def test_combine_refuses_mismatched_step_counts(tmp_path):
    """Split legs run as separate jobs; nothing else forces them to agree.

    Also the regression guard for the openmm leg deriving its step count
    independently (round(n_production_ns * 1e6 / dt)) instead of from
    max(n_steps_list) -- those coincide only at the default --n-production-ns.
    """
    with pytest.raises(RuntimeError, match="step counts do not match"):
        _combine(tmp_path, _prolix_leg(), _openmm_leg(openmm_n_production_steps=1000))


def test_combine_refuses_mismatched_smoke_modes(tmp_path):
    with pytest.raises(RuntimeError, match="smoke modes do not match"):
        _combine(tmp_path, _prolix_leg(smoke=True), _openmm_leg(smoke=False))


@pytest.mark.parametrize(
    ("prolix_backend", "openmm_backend"),
    [("openmm", "openmm"), ("prolix", "prolix"), ("both", "openmm"), (None, "openmm")],
)
def test_combine_refuses_mislabelled_or_swapped_legs(tmp_path, prolix_backend, openmm_backend):
    """Passing the same file twice, or swapping --combine-prolix/--combine-openmm."""
    with pytest.raises(RuntimeError, match="backend field"):
        _combine(
            tmp_path,
            _prolix_leg(backend=prolix_backend),
            _openmm_leg(backend=openmm_backend),
        )
