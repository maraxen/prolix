# Agent Coordination Guide

## Append-Only Task Log

All agents MUST append to `.agent/agent_tasks.jsonl` when:

1. Starting a new task
2. Completing a task
3. Encountering a blocker

## JSON Format

```json
{
  "timestamp": "2025-12-06T15:50:00-05:00",
  "agent_id": "unique_agent_identifier",
  "task": "Short task name",
  "status": "IN_PROGRESS|COMPLETED|BLOCKED",
  "files_in_scope": ["path/to/file1.py", "path/to/file2.py"],
  "summary": "Brief description of what was done or what is blocked"
}
```

## File Locking Convention

Before modifying a file, check `agent_tasks.jsonl` for any `IN_PROGRESS` tasks with that file in scope. If found:

- Wait for task completion, or
- Coordinate with that agent

## Scope Boundaries

| Agent Task | File Scope |
|------------|------------|
| Force Field Assessment | `scripts/convert_all_xmls.py`, `data/force_fields/*` |
| Force Field Extensibility | `proxide/src/proxide/physics/force_fields/*` |
| Code Optimization | `src/prolix/physics/*` |
| Simulation Loop | `src/prolix/simulate.py` |
| Parallel Tempering | `src/prolix/pt/*` |
| Ligand Support | `proxide/src/proxide/md/bridge/core.py` |

## Debugging Artifacts

Simulation outputs (GIFs, trajectories, logs) should be stored in the root `outputs/` directory.

- This directory is excluded from version control via `.gitignore`.
- Agents may generate files in the base directory during active debugging for performance or visibility, but **MUST** move them to `outputs/` once the task is complete.
- For files the agent does not need direct read access to, store them in `outputs/` immediately.

## Conflict Resolution

If two agents need the same file:

1. Post to `agent_tasks.jsonl` with status `BLOCKED`
2. The agent with earlier timestamp has priority
3. Coordinate via task summaries
