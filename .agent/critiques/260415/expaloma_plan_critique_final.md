# Oracle Critique: Expaloma Plan (Final)

**Verdict:** ✅ APPROVE  
**Confidence:** High  
**Approved for Execution:** Yes

## Strategic Assessment

The plan is architecturally sound, correctly scoped across three components, and grounded in verified codebase analysis. The `expaloma` naming prevents import collisions with upstream `espaloma`/`espaloma_charge`. Equinox is the right choice given Prolix already uses it extensively (`padding.py`, `simulate.py`). The `AtomMapNum` defense is well-motivated even though code review of `from_rdkit_mol` confirms the current upstream preserves creation order — defensive coding here is cheap insurance against upstream changes. The `jax.lax.scan` approach for homogeneous GNN layers and `segment_sum` for message passing are idiomatic JAX and align with Prolix conventions.

## Concerns

### 1. Feature vector dimensionality — *suggestion*

**Issue:** The plan specifies `Linear(116, 128)` for input featurization, but upstream `utils.py` produces `h_v` of shape `(n_atoms, 100)` for element one-hot + `(n_atoms, 16)` for fingerprints when `use_fp=True`, totaling 116. If `use_fp=False` it's 100. The expaloma port should document which feature mode it targets and assert the expected input dimension at model init.

**Recommendation:** Add an assertion in `EspalomaChargeModel.__init__` that validates `in_features` against the checkpoint's first layer weight shape. Default to 116 (with fingerprints) matching the upstream default.

### 2. Weight conversion checkpoint path — *suggestion*

**Issue:** The plan says to cache converted weights as `.eqx` alongside the original `.pt`. The upstream model is downloaded to CWD as `.espaloma_charge_model.pt` (a hidden file). The expaloma weight loader should use `XDG_DATA_HOME` or a configurable path, not CWD.

**Recommendation:** Use `appdirs` or a `EXPALOMA_WEIGHTS_DIR` environment variable for the cache location, defaulting to `~/.cache/expaloma/`. This prevents littering the working directory.

## Verdict Rationale

No critical or warning-level concerns. Both suggestions are quality improvements that can be incorporated during execution without plan revision. The plan correctly addresses all prior critique cycles:

1. ✅ JAX idiomatic compilation (bucketing, `lax.scan`, no Python loops in JIT)
2. ✅ Atom index preservation via `AtomMapNum` (verified against `from_rdkit_mol` source)
3. ✅ CI dependency isolation via `pytest.importorskip`
4. ✅ Quantitative deviation bounds vs OpenMM
5. ✅ Namespace collision avoidance (`expaloma` ≠ `espaloma`)
6. ✅ Equinox as NN framework (aligned with existing Prolix usage)

**Ready for execution.**
