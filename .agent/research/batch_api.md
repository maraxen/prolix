# Research: Batch API for Task and Debt Creation

## Objective
Design batch/multiple entry support for `task(action: "create")` and `debt(action: "add")` MCP tools to improve efficiency when creating multiple items at once.

## 1. API Design Options

### Option A: dedicated Batch Tools (Recommended)
Create new tools specifically for batch operations: `task_create_batch` and `debt_add_batch`.
*   **Pros:** Clear separation of concerns, no breaking changes to existing tools, simple schema validation.
*   **Cons:** Increases tool count.

### Option B: Extend Existing Tools (Union Types)
Modify `task(action: "create")` to accept either a single object or an array of objects.
*   **Pros:** Keeps tool count low.
*   **Cons:** Complex schema (oneOf), potentially confusing for LLMs, requires client-side changes to handle polymorphic input if strict typing is used.

### Option C: Transaction Wrapper
Generic `batch_execute` tool.
*   **Pros:** Universal.
*   **Cons:** Too complex for this specific use case, high risk of hallucination/misuse by LLMs.

**Recommendation:** Proceed with **Option A**. It is the safest and most explicit path for LLM interaction.

## 2. Schema Design

We will reuse the existing input structs by wrapping them in a vector.

### Task Batch Schema

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TaskBatchCreateRequest {
    pub tasks: Vec<TaskCreateRequest>,
}
```

### Debt Batch Schema

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DebtBatchAddRequest {
    pub items: Vec<DebtAddRequest>,
}
```

This reuses the existing `TaskCreateRequest` and `DebtAddRequest` definitions, ensuring consistency.

## 3. Error Handling

### Strategy: Atomic Transaction (All-or-Nothing)
For data integrity, it is often better to succeed completely or fail completely. Since we are using a database (SQLite likely, based on `db.rs`), we can wrap the loop in a transaction.

*   **Pros:** No partial state; easier to retry.
*   **Cons:** One bad item blocks the whole batch.

### Strategy: Best Effort (Partial Success)
Process each item individually.
*   **Pros:** Valid items get created.
*   **Cons:** Caller must handle partial failures; response format is more complex.

**Recommendation:** **Atomic Transaction**.
For an AI agent, simple success/failure states are easier to handle than partial recovery. If the agent generates 5 tasks and 1 is malformed, it should fix the malformed one and retry the batch, rather than having to figure out which 4 succeeded and which 1 failed and retry only that one.

## 4. Response Format

The response should return the list of created IDs to allow the agent to reference them immediately if needed.

### Task Batch Response
```json
{
  "count": 3,
  "ids": ["260120100001", "260120100002", "260120100003"],
  "message": "Successfully created 3 tasks"
}
```

### Debt Batch Response
```json
{
  "count": 2,
  "ids": [104, 105], // Auto-increment IDs
  "message": "Successfully added 2 technical debt items"
}
```

## Implementation Plan

1.  **Update `agent-infra-mcp/src/server_tools/tasks.rs`**:
    *   Define `TaskBatchCreateRequest`.
    *   Implement `task_create_batch`.
    *   Use a DB transaction to iterate and insert `tasks`.

2.  **Update `agent-infra-mcp/src/server_tools/debt.rs`**:
    *   Define `DebtBatchAddRequest`.
    *   Implement `debt_add_batch`.
    *   Use a DB transaction to iterate and insert `items`.

3.  **DB Layer**:
    *   Ensure `db.rs` exposes a way to run a transaction or that the insert methods can be composed within a transaction lock.
