# Development Matrix

> Single source of truth for all tasks. Use grep/sed for programmatic updates.
> See `dev-matrix-skill` for interaction patterns.

## Quick Reference

- **Add task**: Use dev-matrix skill or manually add row
- **Update status**: `sed -i '' 's/| {ID} | TODO/| {ID} | IN_PROGRESS/' DEVELOPMENT_MATRIX.md`
- **Find P1 items**: `grep "| TODO | P1 |" DEVELOPMENT_MATRIX.md`

## Task Registry

| ID | Status | Pri | Diff | Mode | Skills | Research | Workflows | Agents | Description | Created | Updated |
|----|--------|-----|------|------|--------|----------|-----------|--------|-------------|---------|---------|
| 000000 | TODO | P1 | med | orchestrator | - | - | - | - | Example task (delete me) | YYMMDD | YYMMDD |

## Status Legend

- `TODO` - Not started
- `IN_PROGRESS` - Active work
- `BLOCKED` - Waiting on dependency
- `REVIEW` - Needs verification
- `DONE` - Complete

## Priority Legend

- `P1` - Critical path, do immediately
- `P2` - Important, do soon
- `P3` - Should do eventually
- `P4` - Nice to have
