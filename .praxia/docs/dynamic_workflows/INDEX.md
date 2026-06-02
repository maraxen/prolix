# Dynamic Workflows

Claude Code `Workflow(...)` scripts for this project. Each script is self-contained,
invocable directly from a Claude Code session, and persisted here for resume/iteration.

Invocation pattern:
```
Workflow({ scriptPath: '.praxia/docs/dynamic_workflows/<file>.js' })
// or with args:
Workflow({ scriptPath: '...', args: { phase: 'verify', oracle_job: '15341583', gate_job: '15343380' } })
```

---

## Phase 5: Settle R-Step Fix

- [260602_phase5-settle-rstep.js](260602_phase5-settle-rstep.js) — Implement and validate OpenMM-style R-step momentum correction in settle_langevin. Two-phase invocation: `impl` (default) runs code work + cluster submit; `verify` (with job IDs) pulls results, runs parity check, and closes Phase 5. Backlog items 856–862. Campaign `ef45b8b4`.
