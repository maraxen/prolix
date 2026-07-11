# Next steps — after B1-SMOKE

**date:** 2026-07-11  
**branch:** `b1/smoke`  
**PR landed:** https://github.com/maraxen/prolix/pull/2 (merged)  
**next epic:** `260528_b1-full`  
**invariants:** `.praxia/loop_priorities.toml`

## Where we are

| Leaf | Status | Gate |
|------|--------|------|
| XA-* audit | **completed** | VERIFY PASS |
| TRIAGE | **completed** | next = B1-full |
| B1-LAND | **completed** | PR #2 merged (admin) |
| **B1-SMOKE** | **completed** | B=4 hetero + AOT-ratio `< 0.5` (`tests/bench/test_b1_smoke.py`) |
| B1-FULL | **ready** | B=64 `bth run` Claim-1 campaign |
| XA-NL-DEBT | **ready** | not on B1 critical path |

```mermaid
flowchart LR
  smokeDone[B1-SMOKE done]
  full[B1-FULL]
  smokeDone --> full
```

## Immediate

1. **B1-FULL** — scaffold/run `scripts/benchmarks/b1_init_exec.py` + bathos sidecar; B=64, 100 ps, H100 primary (prereg 260528). Do not use pytest for cluster runs.
2. Paper / HP4 remain deferred until B1-full headline numbers exist.

## Callouts

- Cite OMM-WATER via `gate_pass` / JSON, not bathos `outcome`.
- Honor VACUUM-DT + `exception_*` invariants.
- B1-smoke stays `@pytest.mark.slow` (nightly); default CI marker unchanged.
