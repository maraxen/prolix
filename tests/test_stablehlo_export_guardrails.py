"""StableHLO export guardrails (Phase 1, task 1.3).

Every energy function must be exportable through ``prolix.export`` as *portable*
StableHLO (GEMINI.md "Export" standard). These fast, CPU-only tests lower the key energy
paths and inspect the StableHLO text for things that would break WASM/WebGPU/IREE export:

* Pallas/Triton/Mosaic kernels (``custom_call`` to a GPU/TPU kernel backend) -- must never
  appear, *even when Pallas is requested*, because ``export_context()`` forces pure-XLA;
* vendor linear-algebra custom calls (``lapack_*``, ``cusolver``, ``cublas``...), which are
  not portable across StableHLO consumers;
* any ``custom_call`` outside a tiny reviewed allowlist.

They also check that gradients (forces) lower cleanly and that a serialised
``jax.export`` artifact round-trips to the same numbers.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix import export
from prolix.pallas_kernels import resolve_pallas_mode
from prolix.physics.flash_explicit import (
    chunked_explicit_nonbonded_energy,
    flash_explicit_energy,
)
from prolix.physics.pme import compute_pme_grid_dims, spme_energy_with_forces

pytestmark = pytest.mark.fast

# Reviewed custom-call targets that are acceptable in portable StableHLO. Keep EMPTY unless a
# StableHLO consumer is known to handle the target; add an entry *with a reason*.
ALLOWED_CUSTOM_CALL_TARGETS: set[str] = set()

# Substrings that indicate a non-portable kernel backend leaked into the lowered module.
FORBIDDEN_SUBSTRINGS = (
    "triton",
    "mosaic",
    "pallas",
    "tpu_custom_call",
    "xla.gpu",
    "lapack",
    "cusolver",
    "cublas",
    "cudnn",
)

_CUSTOM_CALL_RE = re.compile(r"stablehlo\.custom_call\s+@([\w.$\-]+)")
_TARGET_NAME_RE = re.compile(r'call_target_name\s*=\s*"([^"]+)"')


def custom_call_targets(text: str) -> set[str]:
    """All ``custom_call`` targets mentioned in a StableHLO/MLIR module."""
    return set(_CUSTOM_CALL_RE.findall(text)) | set(_TARGET_NAME_RE.findall(text))


def assert_portable(text: str, what: str) -> None:
    low = text.lower()
    leaked = [s for s in FORBIDDEN_SUBSTRINGS if s in low]
    assert not leaked, f"{what}: non-portable backend markers in StableHLO: {leaked}"
    targets = custom_call_targets(text)
    extra = sorted(targets - ALLOWED_CUSTOM_CALL_TARGETS)
    assert not extra, f"{what}: unreviewed custom_call target(s) in StableHLO: {extra}"


def lowered_text(fn, *args) -> str:
    """StableHLO text of ``jit(fn)`` lowered inside ``export_context()``."""
    with export.export_context():
        return jax.jit(fn).lower(*args).as_text()


@pytest.fixture(scope="module")
def padded_sys():
    from tests.physics.test_xtrax_pallas_boundary import _make_minimal_periodic_padded_system

    return _make_minimal_periodic_padded_system(n_real=6, n_padded=8)


# -----------------------------------------------------------------------------
# The gate itself
# -----------------------------------------------------------------------------


def test_export_context_forces_pure_xla_even_when_pallas_requested(monkeypatch):
    monkeypatch.setenv("PROLIX_USE_PALLAS", "1")
    assert resolve_pallas_mode(True)[0] is True  # outside export_context Pallas is honoured
    with export.export_context():
        assert export.in_export_context()
        assert resolve_pallas_mode(True) == (False, False)
        assert resolve_pallas_mode(None) == (False, False)
    assert not export.in_export_context()


def test_portability_checker_detects_forbidden_markers():
    """The guardrail must actually bite: seed it with the things it should reject."""
    with pytest.raises(AssertionError, match="non-portable"):
        assert_portable('stablehlo.custom_call @__gpu$xla.gpu.triton(%0)', "synthetic")
    with pytest.raises(AssertionError, match="non-portable"):
        assert_portable('call_target_name = "lapack_spotrf_ffi"', "synthetic")
    with pytest.raises(AssertionError, match="unreviewed custom_call"):
        assert_portable('stablehlo.custom_call @my_kernel(%0)', "synthetic")
    assert_portable("stablehlo.add %0, %1 : tensor<f32>", "clean")


# -----------------------------------------------------------------------------
# Key energy paths lower to portable StableHLO
# -----------------------------------------------------------------------------


def test_spme_energy_and_grad_lower_portably_with_pallas_requested():
    n = 16
    rng = np.random.default_rng(0)
    pos = jnp.asarray(rng.uniform(1.0, 14.0, (n, 3)).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.5, 0.5, n).astype(np.float32))
    mask = jnp.ones((n,), dtype=bool)
    box = jnp.full((3,), 16.0, dtype=jnp.float32)
    grid = compute_pme_grid_dims(np.asarray(box), grid_spacing=2.0, min_dim=8)

    def e(p):
        return spme_energy_with_forces(p, q, mask, box, grid, 0.34, 4, True)  # use_pallas_pme

    assert_portable(lowered_text(e, pos), "spme energy (use_pallas_pme=True)")
    assert_portable(lowered_text(jax.grad(e), pos), "spme grad (use_pallas_pme=True)")


def test_chunked_explicit_nonbonded_lowers_portably_with_pallas_requested(padded_sys):
    sys = padded_sys

    def e(p):
        s = sys.replace(positions=p)
        return chunked_explicit_nonbonded_energy(
            p, s.charges, s.sigmas, s.epsilons, s.atom_mask, s,
            T=8, pme_alpha=float(sys.pme_alpha), use_pallas=True,
        )

    assert_portable(lowered_text(e, sys.positions), "chunked explicit nonbonded")
    assert_portable(lowered_text(jax.grad(e), sys.positions), "chunked explicit nonbonded grad")


def test_flash_explicit_energy_lowers_portably_with_pallas_requested(padded_sys):
    sys = padded_sys

    def e(p):
        return flash_explicit_energy(sys.replace(positions=p), T=8, use_pallas=True)

    assert_portable(lowered_text(e, sys.positions), "flash_explicit_energy")


def test_batched_nl_energy_and_grad_lower_portably(padded_sys):
    from prolix.batched_energy import single_padded_energy_nl

    sys = padded_sys
    n = sys.n_padded_atoms
    nbr = np.full((n, n), n, dtype=np.int32)
    for i in range(n):
        c = 0
        for j in range(n):
            if i != j:
                nbr[i, c] = j
                c += 1
    nbr = jnp.asarray(nbr)
    disp, _ = space.periodic(jnp.full((3,), 25.0, dtype=jnp.float32))

    def e(p):
        return single_padded_energy_nl(
            sys.replace(positions=p), nbr, disp, implicit_solvent=False
        )

    assert_portable(lowered_text(e, sys.positions), "single_padded_energy_nl")
    assert_portable(lowered_text(jax.grad(e), sys.positions), "single_padded_energy_nl grad")


def test_export_energy_fn_artifact_is_portable_and_matches_jit():
    """``export_energy_fn`` on a tiny explicit-params fn: portable text + numerics match."""
    def fn(params, pos):
        d = pos[:, None, :] - pos[None, :, :]
        r2 = jnp.sum(d * d, axis=-1) + jnp.eye(pos.shape[0])
        return jnp.sum(params * 1.0 / r2)

    pos = jnp.asarray(np.random.default_rng(1).uniform(1, 5, (6, 3)).astype(np.float32))
    params = jnp.ones((6, 6), dtype=jnp.float32)
    lowered = export.export_energy_fn(fn, params, pos)
    assert_portable(lowered.as_text(), "export_energy_fn artifact")
    got = float(lowered.compile()(params, pos))
    want = float(jax.jit(fn)(params, pos))
    assert got == pytest.approx(want, rel=1e-6)


def test_jax_export_serialize_roundtrip_matches(padded_sys):
    """Serialised ``jax.export`` artifact round-trips and reproduces the energy and force."""
    pytest.importorskip("flatbuffers", reason="jax.export serialization needs flatbuffers")
    sys = padded_sys
    n = sys.n_padded_atoms
    nbr = np.full((n, n), n, dtype=np.int32)
    for i in range(n):
        c = 0
        for j in range(n):
            if i != j:
                nbr[i, c] = j
                c += 1
    nbr = jnp.asarray(nbr)
    disp, _ = space.periodic(jnp.full((3,), 25.0, dtype=jnp.float32))
    from prolix.batched_energy import single_padded_energy_nl

    def e(p):
        return single_padded_energy_nl(
            sys.replace(positions=p), nbr, disp, implicit_solvent=False
        )

    vg = jax.value_and_grad(e)
    with export.export_context():
        exp = jax.export.export(jax.jit(vg))(
            jax.ShapeDtypeStruct(sys.positions.shape, sys.positions.dtype)
        )
    blob = exp.serialize()
    assert_portable(exp.mlir_module(), "jax.export module (value_and_grad NL energy)")
    e1, g1 = jax.export.deserialize(blob).call(sys.positions)
    e0, g0 = vg(sys.positions)
    np.testing.assert_allclose(e1, e0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g1, g0, rtol=1e-4, atol=1e-4)
