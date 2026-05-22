# Research: Prompt Versioning and Optimization

## Objective
Establish a system to track the evolution of prompts, manage versions, and empirically measure their effectiveness to drive continuous improvement.

## 1. Versioning Scheme

### Hash-Based vs Semantic
*   **Hash-Based (Git-like):** `prompt_id` is the SHA256 of the content. Good for deduplication.
*   **Semantic (v1.0.0):** Good for human communication ("Use v2").
*   **Sequential (v1, v2, v3):** Simple and effective for linear evolution.

**Recommendation:** **Sequential Integer** per prompt name.
`("code_review", 1)`, `("code_review", 2)`. This is easiest to query ("Give me the latest version").

## 2. Schema Extensions

We need to separate the *definition* of a prompt from its *versions*.

```sql
CREATE TABLE prompt_versions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    changelog TEXT,             -- "Added output schema instructions"
    author TEXT,
    created_at TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 0,
    UNIQUE(name, version)
);
```

The existing `prompts` table can serve as the "Head" pointer (current active version), or we can query `prompt_versions` directly.

## 3. Effectiveness Metrics

To know if v2 is better than v1, we need metrics linked to Dispatches.

*   **Completion Rate:** % of dispatches using this prompt version that ended in `status='completed'` (vs `failed`).
*   **Correction Rate:** How often did the user have to intervene or retry? (Hard to measure automatically unless we track "Retry" actions).
*   **Token Efficiency:** Output tokens per successful task.
*   **Latent Feedback:** Did the user accept the result?

## 4. A/B Testing framework

1.  **Variant Management:** Define `traffic_split` configuration (e.g., 50% v1, 50% v2).
2.  **Dispatch Logic:** When `dispatch(action: "claim")` is called, if the task uses a generic prompt name, the system selects a version based on the split.
3.  **Analysis:** `prompt_metrics` tool compares the success rate of v1 vs v2.

## 5. MCP Tools

### `prompt_update`
Creates a new version.
*   Input: `{ name, content, changelog }`
*   Logic: Increments version number, inserts into `prompt_versions`.

### `prompt_rollback`
Sets an older version as active.

### `prompt_metrics`
Aggregates usage stats.
*   Input: `{ name }`
*   Output: `[{ version: 1, success_rate: 0.9 }, { version: 2, success_rate: 0.95 }]`

## Implementation Plan
1.  Create `prompt_versions` table.
2.  Update `insert_prompt` to append to history.
3.  Update `dispatches` table to track `prompt_version` used (already has `prompt_hash`, which is a form of version tracking, but explicit version numbers are better for analysis).
