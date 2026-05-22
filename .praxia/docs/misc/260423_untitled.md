# Oracle critique record — explicit-solvent execution plan (2026-04-21)

**Artifact:** Execution plan v1→v4 (explicit solvent / TIP3P gates / release path)  
**Workflow:** `.agent/workflows/oracle-critique.md`  
**Plan file:** `.agent/docs/plans/260421.md`

## Cycle summary

| Cycle | Verdict | Bottom line |
|-------|---------|-------------|
| 1 | REVISE | Formal adapter compare vs tightening; release tree vs P2a-B2 non-blocking |
| 2 | REVISE | `export_regression_pme` did not guard benchmark literals; comms misread risk on “Prolix R fail” |
| 3 | REVISE | Prefer PME dedup over triple AST; adapter needs mandatory CI; close D1 |
| 4 | APPROVE | v4 execution-ready; suggestions only (module placement, pytest nodes, doc sweep) |

## Final structured critique (schema)

```json
{
  "verdict": "APPROVE",
  "confidence": "high",
  "strategic_assessment": "v4 closes PME triplication via a single module, forces a binary CI contract for aggregate/adapter gates, versions tightening aggregates, and aligns release language with P2a-B2-R vs cross-engine X. Pointer table matches openmm-nightly; remaining work is implementation detail.",
  "concerns": [
    {
      "area": "PME module placement",
      "severity": "suggestion",
      "issue": "Package module vs tests-local is a packaging preference.",
      "recommendation": "Using src/prolix/physics/regression_explicit_pme.py keeps benchmarks and pytest on one import path; document in export script header."
    },
    {
      "area": "Adapter CI",
      "severity": "suggestion",
      "issue": "Pin exact pytest node IDs and golden fixture paths when W3 lands.",
      "recommendation": "Add workflow comment with measured runtime once aggregate tests exist."
    },
    {
      "area": "Docs drift",
      "severity": "suggestion",
      "issue": "Runbook/protocol may still mention conftest-only SSOT.",
      "recommendation": "Sweep after W2/W3 merges."
    }
  ],
  "approved_for_execution": true
}
```

## Machine-readable twin

`.agent/critiques/oracle_critique_260421.json`
