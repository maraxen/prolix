# Research: Cost and Token Tracking

## Objective
Implement a system to track, analyze, and optimize API usage and costs associated with agent tasks and dispatches.

## 1. Data Sources

### Agent/CLI Metadata
The primary source of truth is the LLM provider's response.
*   **Gemini CLI:** Outputs token usage (prompt/completion) at the end of sessions or in verbose logs.
*   **MCP Tools:** Some tools (like `llm_generate` if it existed) return usage stats.
*   **Self-Reporting:** Agents (like `antigravity`) can be instructed to parse their own usage if accessible and report it back.

### Metrics to Track
*   **Input Tokens (Context):** The cost of reading context.
*   **Output Tokens (Generation):** The cost of "thinking" and writing.
*   **Cache Read/Write Tokens:** If using context caching (Gemini), these have different price points.
*   **Latency:** Time to first token (TTFT) and total duration.
*   **Model ID:** Cost varies significantly by model (Flash vs Pro).

## 2. Schema Design

We should track usage at the lowest granular level (Dispatch) and potentially aggregate it.

### Usage Table (New)

```sql
CREATE TABLE usage_logs (
    id INTEGER PRIMARY KEY,
    dispatch_id TEXT NOT NULL,
    task_id TEXT,               -- Denormalized for easier querying
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    duration_ms INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(dispatch_id) REFERENCES dispatches(id)
);

CREATE INDEX idx_usage_task ON usage_logs(task_id);
CREATE INDEX idx_usage_dispatch ON usage_logs(dispatch_id);
```

This allows multiple usage entries per dispatch (e.g., if a dispatch involves multiple turns or tool calls that are tracked separately).

## 3. Reporting and Aggregation

We need SQL queries or MCP tools to answer:
*   "How much did Task X cost?" (`SELECT SUM(cost_usd) FROM usage_logs WHERE task_id = ?`)
*   "What is the daily spend?" (`GROUP BY created_at DATE`)
*   "Which model is consuming the most budget?"

## 4. MCP Tool Interface

### `usage_report`
Generates a summary of usage.
*   Input: `{ start_date, end_date, group_by: ["task", "model", "day"] }`
*   Output: JSON/Table of costs.

### `budget_set`
Sets a spending limit.
*   Input: `{ scope: "project" | "task", limit_usd: 10.0 }`
*   *Enforcement:* The `dispatch(action: "claim")` tool could check the budget before issuing new work.

### `usage_log_add` (Internal)
Used by agents to report their own usage.
*   Input: `{ dispatch_id, input_tokens, output_tokens, model }`
*   *Logic:* Calculates `cost_usd` based on a lookup table of model prices (which needs to be maintained/config file).

## 5. Implementation Strategy

1.  **Price List:** Create a `config/model_prices.json` file mapping model IDs to cost per 1k tokens.
2.  **Schema:** Add `usage_logs` table.
3.  **Collection:**
    *   *Short term:* Agents parse their session summary and call `usage_log_add`.
    *   *Long term:* The orchestration layer (middleware) captures this automatically from the model provider API.

## Challenges
*   **Missing Data:** If the CLI doesn't expose usage programmatically to the running agent, the agent can't report it. We may need to rely on the *caller* of the CLI to log usage.
