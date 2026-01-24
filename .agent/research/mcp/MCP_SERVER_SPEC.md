# Agent Infra MCP Server Specification

> **Status**: PROPOSED
> **Version**: 1.0.0
> **Date**: 2025-01-20
> **Author**: Oracle Agent

## 1. Overview

The `agent-infra-mcp` server provides a unified interface for automating the Agent Matrix workflow. It abstracts the file-system operations required to manage tasks, skills, and agent dispatching, serving as the "operating system" for the `.agent/` directory structure.

**Primary Goals:**

1. **Matrix Integrity**: Enforce schema and validation on `DEVELOPMENT_MATRIX.md`.
2. **Workflow Automation**: standardize task creation, status updates, and archival.
3. **Context Management**: Dynamically load and format skills and prompt context.
4. **Agent Dispatch**: encapsulate the logic for constructing prompts and invoking the Gemini CLI.

## 2. Tool Inventory

The server exposes the following tools, categorized by function.

### Task Management (Matrix & Filesystem)

| Tool | Description |
|------|-------------|
| `task(action: "create")` | Creates a new task in the matrix and scaffolds the directory structure. |
| `task(action: "update")` | Updates task status, priority, or assigned agents in the matrix. |
| `task(action: "list")` | Queries the matrix for tasks matching specific criteria (e.g., "TODO"). |
| `task(action: "archive")` | Moves a completed task directory to the archive and updates the matrix. |
| `task_get_context` | Retrieves the full context (README, matrix row, research links) for a task. |

### Context & Skills

| Tool | Description |
|------|-------------|
| `skill(action: "load")` | Loads skill documentation from `global_skills/` with requested formatting. |
| `handoff_generate` | Generates a session handoff document based on workspace state. |

### Dispatch & Execution

| Tool | Description |
|------|-------------|
| `dispatch_task` | High-level orchestration tool to assemble context and invoke the CLI. |
| `select_model` | Determines the optimal Gemini model based on task mode and complexity. |
| `format_prompt` | Assembles the standardized prompt components (Agent, Skill, Task). |

## 3. Tool Specifications

### 3.1 task(action: "create")

Combines matrix entry creation with directory scaffolding.

```json
{
  "name": "task(action: "create")",
  "description": "Create a new task in the matrix and filesystem.",
  "parameters": {
    "type": "object",
    "properties": {
      "description": { "type": "string", "description": "Brief description for the matrix." },
      "priority": { "type": "string", "enum": ["P1", "P2", "P3", "P4"], "default": "P2" },
      "difficulty": { "type": "string", "enum": ["easy", "med", "hard"], "default": "med" },
      "mode": { "type": "string", "description": "Target agent mode (e.g., orchestrator)." },
      "skills": { 
        "type": "array", 
        "items": { "type": "string" },
        "description": "List of required skill slugs."
      },
      "slug": { "type": "string", "description": "Optional slug for directory name (auto-generated if omitted)." }
    },
    "required": ["description", "mode"]
  }
}
```

### 3.2 task(action: "update")

Handles updates to the single source of truth (`DEVELOPMENT_MATRIX.md`).

```json
{
  "name": "task(action: "update")",
  "description": "Update task status or metadata in the matrix.",
  "parameters": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "description": "6-char Task ID." },
      "status": { "type": "string", "enum": ["TODO", "IN_PROGRESS", "BLOCKED", "REVIEW", "DONE"] },
      "agent": { "type": "string", "description": "Agent @mention to add." },
      "set_fields": {
        "type": "object",
        "description": "Key-value pairs for other columns (e.g., {'Pri': 'P1'})."
      }
    },
    "required": ["id"]
  }
}
```

### 3.3 task(action: "list")

Renamed from `matrix_query` for consistency.

```json
{
  "name": "task(action: "list")",
  "description": "Query the Development Matrix for tasks.",
  "parameters": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "priority": { "type": "string" },
      "mode": { "type": "string" },
      "limit": { "type": "integer", "default": 10 }
    }
  }
}
```

### 3.4 task(action: "archive")

Moves `DONE` tasks to the archive.

```json
{
  "name": "task(action: "archive")",
  "description": "Moves a completed task directory to archive/ and verifies DONE status.",
  "parameters": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "description": "6-char Task ID." }
    },
    "required": ["id"]
  }
}
```

### 3.5 skill(action: "load")

Injects skill knowledge into prompts.

```json
{
  "name": "skill(action: "load")",
  "description": "Loads skills for agent context injection.",
  "parameters": {
    "type": "object",
    "properties": {
      "skills": {
        "type": "array",
        "items": { "type": "string" }
      },
      "format": {
        "type": "string",
        "enum": ["full", "summary", "instructions_only"],
        "default": "summary"
      }
    },
    "required": ["skills"]
  }
}
```

### 3.6 dispatch_task

The primary entry point for agent automation.

```json
{
  "name": "dispatch_task",
  "description": "Assembles context and dispatches a task to the Gemini CLI.",
  "parameters": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "description": "Task ID from matrix." },
      "mode": { "type": "string", "description": "Agent mode (overrides matrix if provided)." },
      "task_override": { "type": "string", "description": "Specific instruction (defaults to task README)." },
      "context_files": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["id"]
  }
}
```

## 4. Architecture Recommendations

### 4.1 Language & Framework

**Recommendation:** TypeScript (Node.js) using the official Model Context Protocol SDK.

* **Reasoning:**
  * **JSON Schema:** First-class support in TS, essential for tool definitions.
  * **Async I/O:** Efficient handling of file system operations (scaffolding, archiving).
  * **Ecosystem:** Fits the existing `gemini` CLI (likely Node-based or closely adjacent) and allows easy integration with other web-based tools if needed.
  * **Type Safety:** Critical for parsing and maintaining the strict format of `DEVELOPMENT_MATRIX.md`.

### 4.2 State Management

* **Stateless Server:** The server should hold no internal state.
* **Source of Truth:** `DEVELOPMENT_MATRIX.md` is the database. The server reads/writes to this file directly for every operation to ensure consistency.
* **Concurrency:** Use file locking or atomic writes when updating the matrix to prevent corruption from concurrent tool usage (though rare in single-user CLI context).

### 4.3 CLI Integration

* The server should spawn the `gemini` CLI as a child process for `dispatch_task`.
* **Environment Variables:** The server must have access to `GEMINI_MODEL_FAST` and `GEMINI_MODEL_DEEP` to pass them to the CLI.

## 5. Implementation Priority

1. **Phase 1: Matrix Foundations (High Value / Low Risk)**
    * `task(action: "list")`: Essential for finding work.
    * `task(action: "create")`: Standardizes the tricky ID generation and table formatting.
    * `task(action: "update")`: Simplifies status reporting.

2. **Phase 2: Context Automation**
    * `skill(action: "load")`: Reduces token usage by only loading relevant skill sections.
    * `task_get_context`: Prerequisite for effective dispatching.

3. **Phase 3: The "Doer"**
    * `dispatch_task`: The complex logic that ties it all together. Requires Phase 1 & 2 to be robust.

4. **Phase 4: Lifecycle Management**
    * `task(action: "archive")`: Cleanup.
    * `handoff_generate`: Advanced workflow optimization.

## 6. Open Questions

1. **Session Continuation:** How should `dispatch_task` handle long-running sessions?
    * *Proposed:* Accept a `session_id` parameter. If present, use the `--session` flag on the CLI.
2. **Output Parsing:** The CLI output format needs to be strictly defined for `dispatch_task` to programmatically update the matrix (e.g., did it fail? did it finish?).
    * *Proposed:* Force JSON output from the CLI when running in headless mode.
3. **Error Handling:** If `task(action: "create")` generates an ID that collides (rare but possible), should it auto-retry?
    * *Decision:* Yes, built-in retry logic for ID generation up to 3 times.
