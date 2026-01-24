# Matrix Operations Analysis for MCP Integration

This document explores the `DEVELOPMENT_MATRIX.md` structure and CRUD patterns to define requirements for automated MCP tools.

## 1. Schema Definition

The matrix is a Markdown table with the following columns:

| Column | Type | Valid Values / Format | Notes |
|--------|------|-----------------------|-------|
| **ID** | String | 6-char hex string | Unique identifier. Examples: `a1b2c3`, `mcp001`. |
| **Status** | Enum | `TODO`, `IN_PROGRESS`, `BLOCKED`, `REVIEW`, `DONE` | Current state of the task. |
| **Pri** | Enum | `P1`, `P2`, `P3`, `P4` | Priority (P1 is highest). |
| **Diff** | Enum | `easy`, `med`, `hard` | Estimated difficulty. |
| **Mode** | String | Agent mode slug | e.g., `orchestrator`, `fixer`, `explorer`. |
| **Skills** | CSV | Comma-separated slugs | Skills needed from `global_skills/`. |
| **Research** | CSV | Path fragments | Relative to `.agent/research/`. |
| **Workflows** | CSV | Workflow slugs | Relative to `.agent/workflows/`. |
| **Agents** | CSV | `@`-prefixed names | Agents that have worked on this task. |
| **Description** | String | Text | Brief summary of the task. |
| **Created** | Date | `YYMMDD` | Creation timestamp. |
| **Updated** | Date | `YYMMDD` | Last modification timestamp. |

## 2. ID Generation Methods

Current patterns for generating 6-character IDs:

1.  **Timestamp-based (Hex):**
    ```bash
    printf '%06x' $(($(date +%s) % 16777216))
    ```
2.  **MD5 Hash-based (Random):**
    ```bash
    echo -n "$(date +%s%N)" | md5 | head -c 6
    ```

**Requirement for MCP:** The tool should ensure uniqueness by checking the existing matrix before assigning a new ID.

## 3. Current CRUD Patterns

Programmatic updates currently rely on `grep` and `sed`. Note: `sed -i ''` is used for macOS compatibility.

| Operation | Method | Example Pattern |
|-----------|--------|-----------------|
| **Add Task** | Append to file | `echo "| {ID} | TODO | ... |" >> .agent/DEVELOPMENT_MATRIX.md` |
| **Update Status** | `sed` replace | `sed -i '' 's/| {id} | TODO |/| {id} | IN_PROGRESS |/' DEVELOPMENT_MATRIX.md` |
| **Add Agent (First)** | `sed` capture | `sed -i '' 's/| {id} \(.*\)| - |/| {id} \1| @{agent} |/' DEVELOPMENT_MATRIX.md` |
| **Append Agent** | `sed` capture | `sed -i '' 's/| {id} \(.*\)| @\([^|]*\) |/| {id} \1| @\2,@{new_agent} |/' DEVELOPMENT_MATRIX.md` |
| **Update Timestamp** | `sed` replace | (Logic usually involves replacing the last column `| YYMMDD |`) |
| **Query TODO** | `grep` | `grep "\| TODO \|" .agent/DEVELOPMENT_MATRIX.md` |
| **Query by ID** | `grep` | `grep "\| {id} \|" .agent/DEVELOPMENT_MATRIX.md` |

## 4. MCP Tool Specifications

### `matrix_add`
Adds a new entry to the matrix.

**JSON Schema:**
```json
{
  "name": "matrix_add",
  "parameters": {
    "type": "object",
    "properties": {
      "description": { "type": "string" },
      "priority": { "type": "string", "enum": ["P1", "P2", "P3", "P4"] },
      "difficulty": { "type": "string", "enum": ["easy", "med", "hard"] },
      "mode": { "type": "string" },
      "skills": { "type": "array", "items": { "type": "string" } },
      "research": { "type": "array", "items": { "type": "string" } },
      "workflows": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["description", "priority", "mode"]
  }
}
```

### `matrix_update`
Updates status, agents, or other metadata for an existing ID.

**JSON Schema:**
```json
{
  "name": "matrix_update",
  "parameters": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "pattern": "^[a-z0-9]{6}$" },
      "status": { "type": "string", "enum": ["TODO", "IN_PROGRESS", "BLOCKED", "REVIEW", "DONE"] },
      "agent": { "type": "string", "description": "Agent name to add (with or without @)" },
      "set_fields": {
        "type": "object",
        "description": "Arbitrary field updates (Skills, Research, etc.)"
      }
    },
    "required": ["id"]
  }
}
```

### `matrix_query`
Filters and returns matrix rows.

**JSON Schema:**
```json
{
  "name": "matrix_query",
  "parameters": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "priority": { "type": "string" },
      "mode": { "type": "string" },
      "id": { "type": "string" },
      "limit": { "type": "integer", "default": 10 }
    }
  }
}
```

## 5. Validation Requirements

1.  **ID Format:** Must be 6 characters, lowercase alphanumeric.
2.  **ID Uniqueness:** No two tasks should share an ID.
3.  **Enum Constancy:** Status and Priority must match allowed values for reliable `grep` queries.
4.  **No Trailing Spaces:** Comma-separated lists (`Skills`, `Research`, `Workflows`) must not have spaces after commas to avoid breaking simple regex matches.
5.  **Timestamp sync:** Any update to a row MUST update the `Updated` column to the current `YYMMDD`.