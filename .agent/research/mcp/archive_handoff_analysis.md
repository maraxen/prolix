# Archive & Handoff Patterns Analysis

> **Status**: COMPLETED
> **Task**: mcp001
> **Subtask**: mcp_05
> **Agent**: @recon
> **Model**: gemini-3-flash-preview

---

## Archive Trigger Conditions

- **Completion**: The primary trigger for archival is a task directory in `tasks/` reaching `DONE` status.
- **Cleanup**: Archival is recommended immediately upon completion to keep the active `tasks/` directory focused.
- **Retention**: Items in the `archive/` should be kept for at least 30 days before considering permanent deletion.
- **Manual vs. Automated**: Currently a manual "move" operation, identified as a prime candidate for MCP automation.

## Archive Structure

The archive maintains the full fidelity of the original task directory to ensure past decisions and artifacts remain discoverable.

```
.agent/archive/
├── README.md                # Index and retention policy
└── {id}_{name}/             # The full task directory moved from tasks/
    ├── README.md            # Task spec and local matrix
    ├── artifacts/           # Research, logs, and generated files
    └── tracking/            # Time tracking or progress logs (if applicable)
```

## Handoff Template Analysis

The `handoff.md` template is designed for high-density context transfer between different agent types (e.g., from a @recon specialist to a @summarize agent or a @fixer).

Key sections include:
- **Role Continuity**: Tracking `Previous Agent` and `Next Agent` to maintain architectural alignment.
- **Current State Snapshot**: Mandatory reporting on `Build`, `Tests`, and `Lint` status ensures the next agent doesn't inherit a broken environment blindly.
- **Delta Tracking**: Explicit lists of `Files Modified` and `Uncommitted Changes` (via `git status`).
- **Critical Context**: A "Must Know" block for non-obvious gotchas and rationale for key decisions.

## Context Preservation Patterns

- **Must preserve**: Original task IDs, artifact references, and uncommitted state.
- **Can summarize**: Intermediate chat logs or repetitive tool output can be compacted by a `summarize` agent into the `Session Summary` or `Key Decisions`.
- **Cross-session linking**: Handoffs must link back to the parent `Task README` and the global `DEVELOPMENT_MATRIX.md`.

## MCP Tool Specifications

### task(action: "archive")

Automates the movement of a completed task from active to historical storage.

```json
{
  "name": "task(action: "archive")",
  "description": "Moves a completed task directory from tasks/ to archive/ and updates relevant indices.",
  "parameters": {
    "type": "object",
    "properties": {
      "task_id": {
        "type": "string",
        "description": "The unique ID or folder name of the task to archive."
      },
      "confirm_status": {
        "type": "boolean",
        "description": "Verify that the task is marked as DONE in the matrix before moving.",
        "default": true
      }
    },
    "required": ["task_id"]
  }
}
```

### handoff_generate

Bootstraps a handoff document by scraping the current workspace state.

```json
{
  "name": "handoff_generate",
  "description": "Generates a session handoff document based on uncommitted changes and recent file modifications.",
  "parameters": {
    "type": "object",
    "properties": {
      "target_agent": {
        "type": "string",
        "description": "The agent role expected to pick up the task."
      },
      "summary": {
        "type": "string",
        "description": "High-level summary of work performed in the current session."
      }
    },
    "required": ["summary"]
  }
}
```
