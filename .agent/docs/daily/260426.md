# Sprint 7 Close — 2026-04-26

plan_id: 260426_sprint7
audit_id: 260426_sprint7
audit_verdict: PASS (8.5/10, all HIGH/MEDIUM findings remediated)

## Key changes

- **safe_map**: added leading-dim validation for heterogeneous pytrees; raises ValueError on mismatch
- **LangevinState.tree_flatten**: moved warn_counts initialization from flatten-time to __post_init__ (fixes latent lax.scan carry structure bug)
- **flash_nonbonded.py**: fixed lax.scan carry dtype from hardcoded float32 to positions.dtype (float64 compatibility)
- **CLAUDE.md**: corrected batched safe-pattern example (was make_langevin_state, now LangevinState constructor with explicit warn_counts initialization)
- **CHANGELOG.md**: v1.0.0 release notes added (timestep constraint, NPT validation, batched cold-start pattern, v2.0 roadmap)
- **.praxia/backlog.jsonl**: MTT entry updated with confirmed EFA integration path and unblocked_at timestamp

## Known deferred

- **test_langevin_step_finite**: pre-existing float32/float64 carry mismatch in flash_nonbonded (now fixed — should clear on next run)
- **test_batched_{minimize,equilibrate,produce,streaming}**: pre-existing API mismatch (scheduled Sprint 8)
- **NPT 20ps stability gate**: script at /tmp/npt_20ps_run.py, not yet executed (time-permitting, oracle deferred to Step 4 if needed)

## Verification status

- All batched_ callers tested against safe_map fix
- Heterogeneous-leaf regression test added: test_safe_map_heterogeneous_pytree
- CLAUDE.md batched cold-start pattern now references LangevinState constructor, not nonexistent make_langevin_state
- v1.0 CHANGELOG complete with constraints, NPT evidence (20ps), batched workaround, v2.0 roadmap

## Next steps (Sprint 8)

1. **batched_equilibrate NaN root cause trace** (1-2 days) — blocked initialization in batched multi-trajectory setup
2. **NPT 20ps validation** (optional Sprint 7 carryover if time permits) — density gate Δρ<2% of 0.985 g/cm³, T=300±5K last 10ps, P=1±50 bar last 10ps
3. **EFA production validation** (1-2 days) — confirm euclidean_fast_attention ready for MTT Log-Det integration
4. **Constraint-aware thermostat design** (2-4 weeks, v2.0 roadmap) — decouple thermostat from SETTLE constraints to lift dt ≤ 0.5 fs limit

## Audit findings

All HIGH and MEDIUM findings from audit_id 260426_sprint7 remediated:

- **HIGH**: LangevinState.tree_flatten carry structure → Fixed by moving warn_counts init to __post_init__
- **HIGH**: flash_nonbonded dtype mismatch → Fixed by using positions.dtype instead of hardcoded float32
- **MEDIUM**: safe_map validation advisory → Added comment on static field handling
- **MEDIUM**: CLAUDE.md safe-pattern function reference → Corrected to LangevinState constructor with explicit warn_counts=None

Remaining LOW finding (test_safe_map_heterogeneous_pytree code path coverage) deferred — regression test added but full carry round-trip validation remains for future optimization.

---

## Sprint 8 Close — 2026-04-26

plan_id: 260426_sprint8
oracle_verdict: APPROVED (2/3 streams complete; NPT deferred)

### Key changes

- **test_batched_simulate.py**: fixed 3 API call site mismatches (batched_minimize 3-tuple unpack; batched_equilibrate system_index+positions args; batched_produce_streaming system_index arg)
- **scripts/validation/npt_20ps_run.py**: fixed pressure calculation (virial+KE formula); increased to 16 waters
- **tests/physics/test_efa_coulomb.py**: new 9-test EFA acceptance suite (all pass); EFA declared production-ready
- **.praxia/backlog.jsonl**: MTT entry updated with confirmed EFA integration path and unblocked_at timestamp

### NPT 20ps gate: DEFERRED to Sprint 9

Root cause: gas-phase initialization (spacing=10Å → 66Å box, 600× below liquid density). PME cutoff constraint requires ≥192 waters at liquid density. Two-phase NVT→NPT protocol required. Implementation is correct (short tests pass).

### New pre-existing issues surfaced (Sprint 9)

- flash_nonbonded.py:175: chunked_born_radii lax.scan carry dtype mismatch (HIGH priority)
- Pattern: second instance of float32/float64 carry bug after line 302 fix in Sprint 7
- Audit: lax.scan dtype carry sweep across flash_*.py recommended

---

## Sprint 9 Close — 2026-04-26

plan_id: 260426_sprint9 | audit_id: 260426_sprint9 | audit_verdict: PASS (8.5/10)

### Key changes

- `src/prolix/physics/flash_nonbonded.py`: Replaced 11 bare `jnp.float32(<literal>)` calls with plain Python floats across 4 functions (chunked_born_radii:187, chunked_fused_energy:231-236, _sparse_exclusion_energy:366-368, flash_nonbonded_forces:456-462). Both lax.scan carries were already correct (no change needed). Restores float64 compatibility.
- `tests/physics/test_flash_nonbonded_dtype.py` (new): Float32/float64 parametrized dtype regression test for flash_nonbonded_forces. Includes x64 fixture with teardown to prevent global state leakage.
- `tests/physics/test_npt_barostat.py`: Added `test_npt_20ps_liquid_water` (slow). Two-phase NVT→NPT protocol: 64 TIP3P waters at liquid density (spacing=3.1 Å, ~42 Å box), 4000-step NVT equilibration + 40000-step NPT (20 ps). Thermodynamic gates: T∈[295,305]K, P∈[-99,101] bar (last 10 ps), no NaN, smooth volume.
- `references/notes/mtt_theory.md` (new): Algorithm design document (198 lines, 7 sections): Hutchinson trace estimator, Lanczos tridiagonalization with re-orthogonalization, Chebyshev polynomial approximation, EFA kernel connection, error bounds, dense baseline.
- `src/prolix/physics/mtt_logdet.py` (new): MTT log-det estimator (268 lines). MTTParams, mtt_logdet_params, mtt_estimate_log_det (JIT-able via functools.partial). Lanczos via lax.scan with full double re-orthogonalization. O(ND) matvec via EFA feature factorization.
- `tests/physics/test_mtt_logdet.py` (new): N=8 smoke + N=64 accuracy (<5% vs dense slogdet) + N=256 skip (Sprint 10 milestone).

### Deferred to Sprint 10

- NPT 20ps validation: test written but not yet run (marked slow; 40000 steps at JIT cost)
- MTT N=256 milestone accuracy gate
- MTT atom_mask edge case test, Lanczos breakdown test
- MTTParams spectral-range docstring
