---
title: "Query patterns"
description: "bathos sql, campaign review, and compact commands, moved verbatim from CLAUDE.md."
status: reference
date: '261007'
---

### Query patterns

```bash
# Recent runs by campaign tag (bath find can be finicky with multi-tag — use sql)
bth sql "SELECT id, status, tags FROM read_parquet('~/.bth/catalog/runs/prolix/run_*.parquet') WHERE list_contains(tags, 's71-external-baseline') ORDER BY timestamp DESC LIMIT 10"

# Campaign-level review
bth campaign review <campaign_id>

# After many fragments accumulate
bth compact
bth sql "SELECT tool, AVG(per_mol_step_seconds) FROM runs WHERE list_contains(tags, 'external-baseline') GROUP BY tool"
```
