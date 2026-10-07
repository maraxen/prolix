---
title: "Anti-patterns observed (don't repeat)"
description: "bathos provenance failures and the fix for each, moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

### Anti-patterns observed (don't repeat)

| Anti-pattern | What happens | Fix |
|---|---|---|
| Slurm wrapper calls `uv run python foo.py` directly | No provenance, no SLURM_JOB_ID capture, no campaign association — run is invisible to bath | Wrap with `uv run bth run python foo.py --tag ... --campaign ...` |
| Putting `[benchmark]` instead of `[experiment]` in the sidecar | Gate may reject "no [outcomes] section found" depending on bath version | Use `[experiment]` schema (verified) |
| Single campaign-level TOML treated as a sidecar | Gate rejects (wrong file location/structure) | Per-script sidecars + a separate `campaign_design.toml` (non-`.bth.toml` extension) for shared design notes |
| Manual `rsync` of `/outputs/` directly to local instead of bath catalog | Results visible but not tracked; can't query later | Use `bth sync` for catalog; direct rsync only for non-bath artifacts |
| Running smoke without registering a campaign first | Run lands in catalog but `bth campaign review` shows nothing | `bth campaign create` BEFORE the first `bth run` |
| Hand-editing sidecars to remove `is_residual = true` from `[outcomes.fail]` | Gate failure: "exactly one outcome must have is_residual=true" | Keep the residual on the catch-all fail branch |
