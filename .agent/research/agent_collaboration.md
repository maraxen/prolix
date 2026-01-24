# Research: Agent Collaboration Patterns

## Objective
Design workflows and protocols for multi-agent collaboration, specifically focusing on code review and complex task orchestration within an asynchronous MCP-based architecture.

## 1. Collaboration Patterns

### Pattern A: Sequential Pipeline (Chain)
Agent A -> Output -> Agent B -> Output -> Agent C
*   *Use Case:* Feature implementation (Plan -> Implement -> Test).
*   *Implementation:* Each dispatch `depends_on` the previous one.

### Pattern B: Parallel Worker / Map-Reduce
Orchestrator splits task into N sub-tasks -> N Agents work in parallel -> Aggregator synthesizes results.
*   *Use Case:* Batch research, migrating multiple files, running test suites.
*   *Implementation:* One "Fan-out" dispatch creates N child dispatches. One "Fan-in" dispatch waits for all N.

### Pattern C: Specialist Routing
Router agent analyzes request and dispatches to the best expert (e.g., "SQL Expert", "React Expert").
*   *Use Case:* General "fix this bug" request.

### Pattern D: Debate / Review (Adversarial)
Agent A proposes solution -> Agent B critiques -> Agent A refines.
*   *Use Case:* Architecture design, security audit.

## 2. Code Review Workflow

A specialized version of Pattern B (Parallel) + D (Critique).

1.  **Author:** Implements feature, creates Pull Request (or local branch).
2.  **Dispatch:** `request_code_review` is called.
3.  **Orchestrator:** Creates 3 parallel dispatches:
    *   `reviewer_security`: Checks for vulnerabilities.
    *   `reviewer_style`: Checks for linting/conventions.
    *   `reviewer_logic`: Checks for bugs/requirements.
4.  **Reviewers:** Submit structured feedback.
5.  **Aggregator:** Synthesizes feedback into a single report. If "Changes Requested", creates a task for the Author to fix.

## 3. Conflict Resolution

When agents disagree (e.g., Security says "Block" but Logic says "LGTM"):

*   **Voting:** Majority wins? (Too simplistic for code).
*   **Hierarchy:** Security vetoes everything. Style is low priority.
*   **Escalation:** If confidence scores conflict, dispatch to a "Principal Architect" (smarter model) to decide.
*   **Confidence Weighting:** `Result = (VoteA * ConfA) + (VoteB * ConfB)`.

## 4. Communication Protocol

Since agents don't share RAM, they communicate via:

*   **Artifacts:** Files on disk (e.g., `design_doc.md`, `diff.patch`).
*   **Structured JSON:** The `result` field in `dispatch(action: "complete")` should contain strictly formatted JSON, not just text.
*   **State Signals:** DB status flags.

### Proposed Feedback Schema
```json
{
  "reviewer": "security_agent",
  "verdict": "reject",
  "confidence": 0.95,
  "comments": [
    {"file": "auth.rs", "line": 50, "severity": "critical", "message": "SQL Injection vulnerability"}
  ]
}
```

## 5. MCP Tool Extensions

To facilitate this, we need specialized tools:

### `dispatch_parallel`
Creates multiple dispatches at once.
*   Input: `[{ prompt: "...", target: "security" }, { prompt: "...", target: "style" }]`

### `review_submit`
Standardizes the output of a review task.
*   Input: `ReviewSchema` (as above).

### `review_aggregate`
Reads results from multiple dispatch IDs and produces a summary.

## Recommendation
Start with **Sequential Pipelines** for simplicity, then implement **Parallel Review** for the Code Review feature as the first advanced pattern.
