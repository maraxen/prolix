# Task System Automation Analysis

> **Status**: ✅ Completed
> **Task**: mcp001
> **Subtask**: mcp_02
> **Agent**: @explorer
> **Model**: gemini-3-flash-preview

---

## 🔄 Task Lifecycle

The task system follows a strict state-machine pattern anchored by the `DEVELOPMENT_MATRIX.md` (the "Source of Truth") and isolated task directories.

### 1. Creation (Backlog)

- **Action**: Identify a need or pick up a backlog item.
- **Protocol**:
    1. Generate a 6-character hex ID (e.g., `6fcf50`).
    2. Add a `TODO` row to `.agent/DEVELOPMENT_MATRIX.md`.
- **Automation Potential**: High. A tool can handle ID collisions and table formatting.

### 2. Activation (Linking)

- **Action**: Start work on a `TODO` item.
- **Protocol**:
    1. Update status to `IN_PROGRESS` in the matrix.
    2. Add the agent mode (e.g., `@orchestrator`) to the `Agents` column.
    3. Create a directory: `.agent/tasks/{id}_{snake_case_name}/`.
    4. Scaffold `README.md` using the `unified_task.md` template.
- **Automation Potential**: High. Ensures consistent directory naming and template adherence.

### 3. Execution (I-P-E-T)

- **Action**: Work through the task using the **Inspect-Plan-Execute-Test** flow.
- **Protocol**:
  - **Inspect**: Exploration findings logged in `README.md`.
  - **Plan**: Detailed steps and Definition of Done.
  - **Execute**: Incremental work, logging significant milestones in the `Work Log` or `Progress Log`.
  - **Test**: Verification steps and results documented.
- **Automation Potential**: Medium. Tools can assist in "logging" progress or verifying "Definition of Done" checkmarks.

### 4. Completion

- **Action**: Finalize work and verify criteria.
- **Protocol**:
    1. Set matrix status to `DONE`.
    2. Finalize task `README.md` with completion date.
- **Automation Potential**: High. Synchronized status updates.

### 5. Archival

- **Action**: Clean up active workspace.
- **Protocol**:
    1. Move the task directory from `tasks/` to `archive/`.
    2. The matrix entry remains as a permanent searchable record.
- **Automation Potential**: High. Automated cleanup based on age or status.

---

## 📁 Directory Convention

Standardized structure for task units:

```text
.agent/tasks/{id}_{slug}/
├── README.md          # Primary command center (based on unified_task.md)
├── tracking/          # (Optional) Granular progress logs, session states
└── artifacts/         # (Optional) Task-specific outputs, diagrams, research
```

---

## 🛠️ MCP Tool Requirements

The following tools are required for the `agent-infra-mcp` server to automate the task system.

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `task(action: "create")` | `description`, `pri`, `diff`, `mode`, `skills` | `id`, `dir_path` | Adds entry to matrix and creates task directory/README. |
| `task(action: "update")` | `id`, `status`, `fields` (dict) | `success` (bool) | Updates matrix row and optionally the task README status. |
| `task_query` | `status`, `priority`, `query` | `List[Task]` | Searches the matrix for matching tasks (e.g., "all TODO P1"). |
| `task_get_context` | `id` | `full_context` (json) | Returns task README, matrix row, and links to research/skills. |
| `task(action: "archive")` | `id` | `archive_path` | Moves a `DONE` task directory to the archive. |

---

## 📐 JSON Schemas

### `task(action: "create")`

```json
{
  "name": "task(action: "create")",
  "description": "Create a new task in the matrix and filesystem",
  "parameters": {
    "type": "object",
    "properties": {
      "description": { "type": "string", "description": "Brief description for the matrix" },
      "priority": { "enum": ["P1", "P2", "P3", "P4"], "default": "P2" },
      "difficulty": { "enum": ["easy", "med", "hard"], "default": "med" },
      "mode": { "type": "string", "description": "Target agent mode (e.g., orchestrator)" },
      "skills": { "type": "array", "items": { "type": "string" }, "description": "List of required skills" },
      "slug": { "type": "string", "description": "Optional slug for directory name" }
    },
    "required": ["description", "mode"]
  }
}
```

### `task(action: "update")`

```json
{
  "name": "task(action: "update")",
  "description": "Update task status or metadata",
  "parameters": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "description": "6-char Task ID" },
      "status": { "enum": ["TODO", "IN_PROGRESS", "BLOCKED", "REVIEW", "DONE"] },
      "agent": { "type": "string", "description": "Agent @mention to add" },
      "priority": { "enum": ["P1", "P2", "P3", "P4"] }
    },
    "required": ["id"]
  }
}
```

### `task_query`

```json
{
  "name": "task_query",
  "description": "Query the Development Matrix for tasks",
  "parameters": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "priority": { "type": "string" },
      "agent": { "type": "string" },
      "skill": { "type": "string" }
    }
  }
}
```
