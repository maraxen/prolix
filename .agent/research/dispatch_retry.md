# Research: Dispatch Retry and Recovery

## Objective
Design robust patterns for handling failed dispatches to ensure system reliability and resilience.

## 1. Failure Taxonomy

Understanding *why* a dispatch failed is crucial for deciding *how* to retry it.

*   **Transient Failures:** Network glitches, API rate limits (429), temporary service unavailability (503).
    *   *Action:* Retry automatically.
*   **Permanent Failures:** Invalid prompt, auth error (401/403), context length exceeded (400), malformed tool arguments.
    *   *Action:* Do not retry. Fail immediately. Move to DLQ.
*   **Timeouts:** Agent took too long to respond.
    *   *Action:* Retry (might be a slow model or network), but with a limit.
*   **Logic/Task Failures:** The agent ran but couldn't complete the objective (e.g., "File not found").
    *   *Action:* Depends on task. Usually requires human intervention or a smarter agent (Self-Correction).

## 2. Retry Strategies

### Exponential Backoff
For transient errors, wait `base * 2^attempt` seconds before retrying.
*   *Implementation:* Requires tracking `attempt_count` and `next_retry_at` timestamp.

### Max Retries
Hard limit (e.g., 3 or 5) to prevent infinite loops.

### Model Fallback
If a complex model (e.g., Claude 3.5 Sonnet) fails or times out, fallback to a faster/cheaper model (e.g., Gemini Flash) if the task allows, or vice-versa (fallback to a smarter model if the fast one fails logic).

## 3. State Recovery & Checkpoints

### Idempotency
Ensure that running the same dispatch twice doesn't cause side effects (e.g., creating duplicate tickets).
*   *Design:* Tools should be idempotent (e.g., `create_or_update` instead of just `create`).

### Checkpointing
For long-running tasks, the agent should save intermediate progress (e.g., to a "scratchpad" or the `result` field) so a retry can resume rather than restart.

## 4. Dead Letter Queue (DLQ)

When a dispatch exceeds max retries or encounters a permanent error, it shouldn't just vanish or block the queue.

### Schema Update
Mark status as `dead_letter` or `failed`.
Add `error_log` column to store the history of failure reasons.

### Manual Review Workflow
1.  User runs `dispatch list --status failed`.
2.  User inspects error log.
3.  User can:
    *   **Retry:** `dispatch retry <id>` (resets attempts, sets status to pending).
    *   **Edit:** `dispatch edit <id>` (fix the prompt) then retry.
    *   **Delete:** `dispatch delete <id>` (give up).

## 5. Monitoring & Alerting

*   **Metrics:**
    *   `dispatch_success_rate`: % of tasks completed vs failed.
    *   `retry_rate`: % of tasks requiring >1 attempt.
    *   `queue_depth`: Number of pending tasks (lag).
*   **Alerting:**
    *   If `queue_depth` > X (backlog growing).
    *   If `dispatch_success_rate` < 90% (system stability issue).

## Implementation Plan: Schema Changes

We need to update the `dispatches` table schema to support these features.

```sql
ALTER TABLE dispatches ADD COLUMN attempt_count INTEGER DEFAULT 0;
ALTER TABLE dispatches ADD COLUMN max_retries INTEGER DEFAULT 3;
ALTER TABLE dispatches ADD COLUMN last_error TEXT;
ALTER TABLE dispatches ADD COLUMN next_retry_at TEXT; -- ISO8601 timestamp
ALTER TABLE dispatches ADD COLUMN priority INTEGER DEFAULT 0; -- Higher runs first
```

### Logic Changes in `dispatch(action: "claim")`
1.  Select tasks where `status = 'pending'` OR (`status = 'retrying' AND next_retry_at <= NOW()`).
2.  Increment `attempt_count` upon claim.

### Logic Changes in `dispatch(action: "complete")`
1.  If status is `failed`:
    *   If `attempt_count < max_retries`:
        *   Set status = `retrying`.
        *   Set `next_retry_at` = NOW + backoff.
        *   Update `last_error`.
    *   Else:
        *   Set status = `failed` (DLQ).
