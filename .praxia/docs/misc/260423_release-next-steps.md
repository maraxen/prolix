# Prolix v1.0 Release — Next Steps After Validation

## Current Status (2026-04-23)

**Phase 2 Investigation**: COMPLETE  
**Oracle Decision**: APPROVED (Option B: SETTLE + Langevin + dt ≤ 0.5fs)  
**Validation**: In progress (dt=0.5fs test for 50+ ps)

---

## Validation Gate

Test: `scripts/validate_langevin_settle_dt05fs.py`  
Configuration: SETTLE + Langevin at dt=0.5fs, 100ps, 2 waters  
Gate Criteria: Mean T = 300 ± 5K  

**Outcomes**:

### ✓ If Validation PASSES (T within ±5K)
Proceed immediately to v1.0 release:

1. **Documentation Update** (30 min)
   - ✓ CLAUDE.md created (Phase 2 constraint documented)
   - ✓ settle.py docstring updated (dt ≤ 0.5fs note added)
   - [ ] Release notes: Add Phase 2 timestep limitation
   - [ ] Design docs: Archive Phase 2 investigation findings

2. **Regression Testing** (1-2 hours)
   ```bash
   uv run pytest tests/physics/ -x -q
   ```
   - Verify Phase 1 implicit solvent still works
   - Verify Phase 2 explicit water temperature control
   - No regressions expected (using validated code)

3. **Final Preparation** (30 min)
   - [ ] Update version to v1.0
   - [ ] Create release branch: `release/v1.0`
   - [ ] Tag: `v1.0-stable`
   - [ ] Update CHANGELOG with Phase 2 constraint note

4. **Release** (30 min)
   - [ ] Build and test package
   - [ ] Create GitHub release with notes
   - [ ] Point main to stable
   - [ ] Announce v1.0 with "Phase 2 ready, dt ≤ 0.5fs constraint"

**Total time: ~3-4 hours**

### ✗ If Validation FAILS (T diverges or exceeds ±5K)
Escalate to user for decision:

1. **Debug** (1-2 hours)
   - Check settle.py implementation
   - Verify water indices and masses
   - Compare against Phase 2C test configuration
   - May indicate deeper issue with Langevin+SETTLE coupling

2. **Options**
   - **Option A**: Reduce dt further (0.25fs) and retest
   - **Option B**: Ship v1.0-implicit only (Phase 1), defer Phase 2 to v0.3.0
   - **Option C**: Continue investigating constraint-aware thermostat (timeline impact)

---

## Post-Validation Timeline

### If PASS → v1.0 Release (Same Week)
- **Today (2026-04-23)**: Validation + regression tests
- **2026-04-24**: Release preparation + announcement
- **2026-04-25**: v1.0 tagged and published

### If FAIL → Escalation (Options Discussed with User)
- Reassess Phase 2 integration strategy
- May defer explicit water to v0.3.0
- Pivot to alternative approach (constraints-aware thermostat)

---

## Commit Messages (Post-Validation)

When committing release changes:

```
docs(release): v1.0 with Phase 2 explicit solvent + dt≤0.5fs constraint

Phase 2 integration required explicit constraint on timestep (dt ≤ 0.5fs) to 
maintain temperature stability with SETTLE rigid water constraints. Investigation 
found SETTLE velocity constraints couple with thermostat feedback at larger 
timesteps. Solution: use proven Langevin thermostat at reduced timestep.

Explicit water (TIP3P + SETTLE + PME) now supported in v1.0.
Future versions will implement constraint-aware thermostats to remove timestep limitation.

Changes:
- settle_langevin() docstring: document dt ≤ 0.5fs requirement
- CLAUDE.md: Phase 2 constraint explanation and production usage
- validate_langevin_settle_dt05fs.py: validation test confirming stability

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
```

---

## Contingency: If Validation Fails

Fallback release strategy (if dt=0.5fs doesn't work):

**Option: Release v1.0-implicit-only (Phase 1)**
- No Phase 2 code in main repository
- Explicit solvent research preserved in branches
- Clear message: "v1.0 focuses on implicit solvent; explicit solvent coming in v0.3.0"
- Timeline: Still 1 week (clean release)

**Why this works**:
- Phase 1 (implicit solvent + BAOAB Langevin) is fully validated
- Removes Phase 2 uncertainty from release
- Allows more time for Phase 2 constraint-aware thermostat research
- Clear scope for v1.0

---

## Files Modified in This Session

1. **CLAUDE.md** (new)
   - Phase 2 constraint explanation
   - Production usage examples
   - Link to design docs

2. **settle.py** (~line 551)
   - Docstring update: dt ≤ 0.5fs note
   - Implementation unchanged

3. **scripts/validate_langevin_settle_dt05fs.py** (new)
   - Validation test for dt=0.5fs stability
   - 50,000 steps (100ps), 2 waters

4. **.agent/docs/RELEASE_DECISION_v1.0.md** (new)
   - Complete Phase 2 analysis
   - Oracle recommendation with rationale
   - Path forward for v2.0

5. **Memory files**
   - phase2_oracle_recommendation.md: Decision tracking
   - phase2_final_failure_analysis.md: Investigation archive

---

## How to Monitor Progress

**Check validation output**:
```bash
tail -f /tmp/claude-1000/-home-marielle-projects-prolix/cfe1d995-c278-447c-bdc4-b7f499347e49/tasks/bxk4b00n6.output
```

**When complete, check results**:
```bash
grep -E "Mean T:|Status:" /tmp/claude-1000/-home-marielle-projects-prolix/cfe1d995-c278-447c-bdc4-b7f499347e49/tasks/bxk4b00n6.output
```

**Expected output format**:
```
Mean T: XXX.X K (std ±YYY.Y K)
Target: 300 ± 5.0K
Offset: ZZZ.ZK
Status: ✓ PASS  (if T_offset ≤ 5.0K)
        or
Status: ✗ FAIL  (if T_offset > 5.0K)
```

---

## Success Criteria Recap

✓ **v1.0 Release Ready If**:
- dt=0.5fs validation passes (T within ±5K for 50+ ps)
- All Phase 1 regression tests still pass
- Documentation updated with dt ≤ 0.5fs constraint
- No code changes needed (uses validated Phase 1 integrator)

✗ **Escalate If**:
- Validation fails (T unstable at dt=0.5fs)
- Suggests deeper issue with Langevin+SETTLE coupling
- May need to ship v1.0-implicit-only instead

---

## Key Decision Point

This validation test will determine the v1.0 scope:
- **PASS** → Explicit water (Phase 2) ships with timestep constraint
- **FAIL** → Implicit solvent only (Phase 1) ships, defer Phase 2

Either way, v1.0 is a solid, validated release.
