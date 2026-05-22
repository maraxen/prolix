# Research: refresh_context Feature

**Date**: 2026-01-21  
**Researcher**: Antigravity Agent  
**Task ID**: 260120204820  
**Dispatch ID**: d260120210458

---

## Executive Summary

This research proposes a `refresh_context` MCP tool that provides intelligent session recovery and continuation support for AI agents. The feature addresses the critical need for agents to resume work after session timeouts, agent handoffs, or task switching by generating comprehensive summaries and self-contained continuation prompts.

**Key Design Principles**:

1. **Automatic Context Assembly**: Aggregates multi-source context (tasks, dispatches, git state, open files, tech debt)
2. **Verbosity Control**: Supports multiple output formats (minimal, standard, detailed) to optimize token usage
3. **Self-Contained Prompts**: Generates continuation prompts that include all necessary context snippets
4. **Session-Aware**: Tracks session state and detects context drift
5. **Integration-Ready**: Designed to work seamlessly with existing MCP tools and orchestration workflows

---

## 1. Information Sources

The `refresh_context` tool should aggregate data from multiple sources to provide comprehensive session context:

### 1.1 Database Sources (Primary)

**Tasks Table** (`agent.db`)

- **What**: Current task status, priority, difficulty, mode, description
- **Why**: Provides the "what am I working on?" context
- **Query**: Active tasks (status != 'ARCHIVED' AND status != 'DONE')
- **Fields**: id, status, priority, difficulty, mode, description, created_at, updated_at
- **Associations**: skills, research, workflows, agents (via junction tables)

**Dispatches Table** (`agent.db`)

- **What**: Recent dispatch history, execution status, results
- **Why**: Shows what work has been attempted, what succeeded/failed
- **Query**: Last N dispatches (default: 10), optionally filtered by task_id
- **Fields**: id, target, task_id, model, mode, prompt_preview, status, created_at, completed_at, claimed_by, result
- **Use Case**: Detect if previous agent attempts failed, identify retry scenarios

**Tech Debt Table** (`agent.db`)

- **What**: Known issues, gotchas, architectural constraints
- **Why**: Prevents agents from repeating known mistakes or hitting documented blockers
- **Query**: Open tech debt items (status = 'open')
- **Fields**: id, title, category, description, impact, proposed_solution, related_task_id
- **Use Case**: Surface relevant blockers before resuming work

**Research/Recon Tables** (`agent.db`)

- **What**: Structured findings from previous research/reconnaissance
- **Why**: Provides domain knowledge accumulated during the project
- **Query**: Recent reports, optionally filtered by task_id or topic
- **Fields**: topic/target, summary, findings (JSON), sources/recommendations (JSON)
- **Use Case**: Avoid redundant research, leverage prior discoveries

### 1.2 File System Sources (Secondary)

**Git Status** (`.git/`)

- **What**: Uncommitted changes, current branch, recent commits
- **Why**: Shows what code has been modified since last session
- **Commands**:
  - `git status --short` (staged/unstaged changes)
  - `git diff --stat` (change summary)
  - `git log -n 5 --oneline` (recent commits)
  - `git branch --show-current` (active branch)
- **Use Case**: Detect if work is in progress, identify merge conflicts

**Open Files** (from session state, if available)

- **What**: Files that were being edited in the previous session
- **Why**: Indicates where the agent was actively working
- **Source**: Could be passed as input parameter or inferred from recent git changes
- **Use Case**: Resume editing at the exact point of interruption

**Task Directory** (`.agent/tasks/{id}_*/`)

- **What**: Task-specific artifacts, tracking logs, plans
- **Why**: Contains detailed context for the current task
- **Files**:
  - `README.md` (task prompt and status)
  - `tracking/*.md` (incremental progress logs)
  - `artifacts/*` (generated designs, discovery docs)
- **Use Case**: Load task-specific context that may not be in the database

**Workflows** (`.agent/workflows/*.md`)

- **What**: Process definitions and orchestration patterns
- **Why**: Indicates which workflow the agent should follow
- **Query**: List available workflows, optionally load specific workflow content
- **Use Case**: Resume multi-step workflows at the correct phase

### 1.3 Runtime Sources (Tertiary)

**Session Metadata** (if tracked)

- **What**: Session start time, agent mode, model used, token usage
- **Why**: Helps detect session timeouts and context drift
- **Storage**: Could be stored in a new `sessions` table or passed as parameters
- **Use Case**: Determine if context is stale and needs full refresh

**Environment State**

- **What**: Current working directory, environment variables, running processes
- **Why**: Detects if the agent is in the correct execution context
- **Commands**: `pwd`, `env | grep PROJECT`, `ps aux | grep relevant_process`
- **Use Case**: Verify the agent is in the correct workspace before resuming

---

## 2. Summary Format Design

The `refresh_context` tool should support multiple verbosity levels to balance comprehensiveness with token efficiency:

### 2.1 Minimal Format (Token-Optimized)

**Target Audience**: Quick status checks, frequent polling  
**Token Budget**: ~500-1000 tokens  
**Use Case**: "Am I still on track?" checks during long-running workflows

**Structure**:

```json
{
  "workspace": "/path/to/project",
  "current_task": {
    "id": "260120204820",
    "status": "IN_PROGRESS",
    "description": "Research: refresh_context feature"
  },
  "recent_activity": {
    "last_dispatch": "d260120210458 (running)",
    "last_commit": "3 hours ago",
    "uncommitted_changes": 5
  },
  "blockers": [
    "Tech debt #42: Schema validation error in report(action: "research")"
  ],
  "next_action": "Continue research, create markdown report"
}
```

### 2.2 Standard Format (Balanced)

**Target Audience**: Session recovery after timeout, agent handoff  
**Token Budget**: ~2000-4000 tokens  
**Use Case**: Default format for most continuation scenarios

**Structure**:

```json
{
  "workspace": {
    "path": "/path/to/project",
    "workspace_id": "dac63eee-f789-45a1-939a-83e6efad8eb6",
    "branch": "feature/refresh-context"
  },
  "current_task": {
    "id": "260120204820",
    "status": "IN_PROGRESS",
    "priority": "P1",
    "difficulty": "med",
    "mode": "researcher",
    "description": "Research: refresh_context feature - summary and continuation prompt design",
    "created_at": "2026-01-20",
    "skills": [],
    "research": [],
    "workflows": []
  },
  "recent_dispatches": [
    {
      "id": "d260120210458",
      "task_id": "260120204820",
      "status": "running",
      "claimed_by": "claude-sonnet-4",
      "claimed_at": "2026-01-21 02:20:57",
      "prompt_preview": "# Research: refresh_context Feature..."
    }
  ],
  "git_status": {
    "branch": "feature/refresh-context",
    "uncommitted_changes": 5,
    "staged_files": 2,
    "recent_commits": [
      "abc123f Implement dispatch(action: "claim") workflow",
      "def456a Add workspace handshake protocol"
    ]
  },
  "open_tech_debt": [
    {
      "id": 42,
      "title": "MCP schema 'findings' validation error",
      "category": "bug",
      "impact": "Blocks report(action: "research") tool usage"
    }
  ],
  "continuation_prompt": "You were researching the refresh_context feature (task 260120204820). The research report should be created at /path/to/project/.agent/research/refresh_context_feature.md. Focus on: 1) Information sources, 2) Summary format design, 3) Continuation prompt design, 4) MCP tool interface, 5) Use cases."
}
```

### 2.3 Detailed Format (Comprehensive)

**Target Audience**: Complex task resumption, debugging session failures  
**Token Budget**: ~5000-10000 tokens  
**Use Case**: When full context is needed (e.g., after multiple failed attempts)

**Structure**: Extends standard format with:

- Full task directory contents (README.md, tracking logs)
- Complete git diff output (not just stats)
- All related tech debt items (not just blockers)
- Full dispatch history for the current task
- Loaded workflow content (if applicable)
- Recent research/recon findings (full JSON)

**Additional Fields**:

```json
{
  // ... all standard fields ...
  "task_artifacts": {
    "readme": "# Task 260120204820...",
    "tracking_logs": ["log1.md", "log2.md"],
    "artifacts": ["plan.md", "findings.json"]
  },
  "git_diff": "diff --git a/file.rs b/file.rs\n...",
  "related_research": [
    {
      "topic": "Maintenance Prompt Library",
      "summary": "Comprehensive catalog of 47+ maintenance prompts...",
      "findings": { /* full JSON */ }
    }
  ],
  "workflow_context": {
    "active_workflow": "orchestrator",
    "current_phase": "Phase 2: Research Tasks",
    "content": "# Orchestrator Workflow\n..."
  }
}
```

### 2.4 Narrative Format (Human-Readable)

**Target Audience**: Human review, debugging, documentation  
**Token Budget**: ~3000-6000 tokens  
**Use Case**: When a human needs to understand what the agent was doing

**Structure** (Markdown):

```markdown
# Session Context Summary

**Workspace**: `/path/to/project`  
**Generated**: 2026-01-21 02:30:00  
**Session Duration**: 2 hours 15 minutes

## Current Task

You are working on **Task 260120204820** (Priority: P1, Difficulty: Medium)

> Research: refresh_context feature - summary and continuation prompt design

**Status**: IN_PROGRESS  
**Mode**: researcher  
**Created**: 2026-01-20

## Recent Activity

- **Last Dispatch**: d260120210458 (running, claimed by claude-sonnet-4)
- **Last Commit**: 3 hours ago ("Implement dispatch(action: "claim") workflow")
- **Uncommitted Changes**: 5 files modified, 2 staged

## Git Status

**Branch**: `feature/refresh-context`

**Recent Commits**:
1. abc123f Implement dispatch(action: "claim") workflow
2. def456a Add workspace handshake protocol
3. ghi789b Refactor server tools into modules

**Modified Files**:
- `.agent/research/refresh_context_feature.md` (new)
- `agent-infra-mcp/src/db.rs` (+120 -30)
- `agent-infra-mcp/src/server.rs` (+45 -10)

## Known Blockers

⚠️ **Tech Debt #42**: MCP schema 'findings' validation error  
- **Impact**: Blocks report(action: "research") tool usage
- **Category**: bug
- **Proposed Solution**: Reproduce error, identify root cause in schema validation

## Next Steps

Continue the research task. The report should be created at:
`.agent/research/refresh_context_feature.md`

Focus on these research areas:
1. Information sources (database, git, file system)
2. Summary format design (minimal, standard, detailed, narrative)
3. Continuation prompt design (self-contained, context snippets)
4. MCP tool interface (input parameters, output format)
5. Use cases (timeout recovery, agent handoff, task switching)
```

---

## 3. Continuation Prompt Design

The continuation prompt should be **self-contained** and include all necessary context snippets to resume work without requiring the agent to re-query multiple sources.

### 3.1 Prompt Structure

**Template**:

```
# Session Continuation

## Context Restoration

You are resuming work on **{task_description}** (Task ID: {task_id}).

**Previous Session Summary**:
{session_summary}

**Current State**:
- Status: {task_status}
- Priority: {task_priority}
- Mode: {task_mode}
- Last Activity: {last_activity_timestamp}

## Work In Progress

**Git Status**:
- Branch: {current_branch}
- Uncommitted Changes: {uncommitted_count} files
- Staged Files: {staged_count} files

**Modified Files**:
{modified_files_list}

**Recent Commits**:
{recent_commits_list}

## Task Context

**Objective**: {task_description}

**Research Areas** (from task prompt):
{research_areas_or_requirements}

**Expected Output**: {expected_output_location_and_format}

## Known Constraints

**Tech Debt / Blockers**:
{relevant_tech_debt_items}

**Dependencies**:
{task_dependencies_or_prerequisites}

## Continuation Instructions

{specific_next_steps_based_on_current_state}

**Verification Checklist**:
{checklist_items_to_verify_completion}
```

### 3.2 Context Snippet Inclusion Strategy

**Principle**: Include just enough context to avoid re-querying, but not so much that the prompt becomes bloated.

**Inclusion Rules**:

1. **Always Include**:
   - Task ID, description, status, priority, mode
   - Current git branch and uncommitted change count
   - Last dispatch status and result (if any)
   - Critical tech debt items (related_task_id matches or high impact)

2. **Conditionally Include**:
   - Modified files list (if < 20 files, otherwise summarize)
   - Recent commits (last 3-5, with messages)
   - Task directory README.md (if exists and < 2000 tokens)
   - Workflow content (if task has associated workflow and < 3000 tokens)

3. **Never Include** (query on-demand instead):
   - Full git diffs (too large, query if needed)
   - Complete research findings (summarize, link to full report)
   - All tech debt items (filter to relevant only)
   - Historical dispatch logs (only most recent)

### 3.3 Adaptive Continuation Prompts

The continuation prompt should adapt based on the **current state**:

**State: Task Just Started**

```
You are starting work on {task_description}. This is a fresh task with no prior progress.

**First Steps**:
1. Review the task requirements in detail
2. Check for related research or tech debt
3. Create a plan or outline
4. Begin implementation/research
```

**State: Task In Progress, No Recent Activity**

```
You are resuming work on {task_description} after a break. Last activity was {time_ago}.

**Context Refresh**:
- Previous work: {summary_of_previous_work}
- Current state: {current_file_or_artifact_state}

**Resume From**:
{specific_point_to_resume_from}
```

**State: Task In Progress, Recent Failure**

```
You are resuming work on {task_description} after a failed attempt.

**Previous Attempt**:
- Dispatch: {failed_dispatch_id}
- Error: {error_message_or_result}
- Attempted: {what_was_attempted}

**Recovery Strategy**:
{suggested_approach_to_fix_the_issue}
```

**State: Task Blocked**

```
You are resuming work on {task_description}, but there are blockers.

**Blockers**:
{list_of_blocking_tech_debt_or_dependencies}

**Options**:
1. Work around the blocker (if possible)
2. Address the blocker first (if within scope)
3. Escalate to orchestrator (if blocker is external)
```

---

## 4. MCP Tool Interface

### 4.1 Tool Definition

**Tool Name**: `refresh_context`

**Description**: Generate a comprehensive session summary and continuation prompt for resuming work after timeouts, agent handoffs, or task switching.

**Category**: Orchestration / Session Management

### 4.2 Input Parameters

```rust
#[derive(Debug, serde::Deserialize)]
pub struct RefreshContextRequest {
    /// Optional task ID to focus on (defaults to most recent active task)
    pub task_id: Option<String>,
    
    /// Verbosity level: "minimal", "standard", "detailed", "narrative"
    /// Default: "standard"
    pub format: Option<String>,
    
    /// Include git status and diff information
    /// Default: true
    pub include_git: Option<bool>,
    
    /// Include open tech debt items
    /// Default: true
    pub include_tech_debt: Option<bool>,
    
    /// Include recent dispatch history
    /// Default: true
    pub include_dispatches: Option<bool>,
    
    /// Include task directory artifacts (README, tracking logs)
    /// Default: false (only in "detailed" format)
    pub include_task_artifacts: Option<bool>,
    
    /// Include related research/recon findings
    /// Default: false (only in "detailed" format)
    pub include_research: Option<bool>,
    
    /// Number of recent dispatches to include
    /// Default: 5
    pub dispatch_limit: Option<usize>,
    
    /// Number of recent commits to include
    /// Default: 5
    pub commit_limit: Option<usize>,
    
    /// Generate continuation prompt
    /// Default: true
    pub generate_continuation_prompt: Option<bool>,
}
```

### 4.3 Output Format

**Success Response** (JSON):

```json
{
  "workspace": {
    "path": "/path/to/project",
    "workspace_id": "uuid",
    "branch": "branch-name"
  },
  "current_task": { /* DbTask */ },
  "recent_dispatches": [ /* DbDispatch[] */ ],
  "git_status": {
    "branch": "branch-name",
    "uncommitted_changes": 5,
    "staged_files": 2,
    "recent_commits": [ /* commit list */ ]
  },
  "open_tech_debt": [ /* DbTechDebt[] */ ],
  "task_artifacts": { /* optional */ },
  "related_research": [ /* optional */ ],
  "continuation_prompt": "# Session Continuation\n...",
  "metadata": {
    "generated_at": "2026-01-21T02:30:00Z",
    "format": "standard",
    "token_estimate": 3500
  }
}
```

**Error Response**:

```json
{
  "error": "No active workspace. Call workspace_handshake first.",
  "code": "WORKSPACE_NOT_SET"
}
```

### 4.4 Related Tools

The `refresh_context` tool should integrate with existing MCP tools:

**Upstream Tools** (provide data to refresh_context):

- `workspace_handshake`: Establishes the active workspace
- `task(action: "list")`: Queries active tasks
- `dispatch(action: "status")`: Queries recent dispatches
- `debt(action: "list")`: Queries open tech debt
- `prep_orchestrator`: Similar aggregation pattern (can share logic)

**Downstream Tools** (consume refresh_context output):

- `dispatch`: Can use continuation prompt as the dispatch prompt
- `task(action: "update")`: Can update task status based on context
- `dispatch(action: "complete")`: Can include context summary in result

**Composition Pattern**:

```
workspace_handshake() 
  → refresh_context(task_id="abc123", format="standard")
  → dispatch(target="cli", prompt=continuation_prompt)
```

---

## 5. Use Cases

### 5.1 Session Timeout Recovery

**Scenario**: Agent session times out after 2 hours of work on a research task.

**Workflow**:

1. New session starts
2. Agent calls `workspace_handshake(project_root="/path/to/project")`
3. Agent calls `refresh_context(format="standard")`
4. Agent receives summary showing:
   - Task 260120204820 was IN_PROGRESS
   - Last dispatch was "running" (likely timed out)
   - 5 uncommitted files (research report partially written)
5. Agent uses continuation prompt to resume writing the report

**Expected Outcome**: Agent seamlessly continues writing the research report from where it left off, without re-reading all source files.

### 5.2 Agent Handoff

**Scenario**: Claude agent starts a task, but Gemini agent needs to finish it due to model-specific capabilities.

**Workflow**:

1. Claude agent works on task, creates artifacts
2. Claude agent completes dispatch with result summary
3. Gemini agent starts new session
4. Gemini agent calls `refresh_context(task_id="abc123", format="detailed")`
5. Gemini agent receives:
   - Full task context
   - Claude's dispatch result
   - All artifacts created by Claude
   - Continuation prompt explaining what Claude did and what's left
6. Gemini agent continues from where Claude left off

**Expected Outcome**: Gemini agent understands Claude's work and completes the task without duplicating effort.

### 5.3 Task Switching

**Scenario**: Agent is working on Task A, but needs to switch to high-priority Task B, then return to Task A.

**Workflow**:

1. Agent working on Task A (status: IN_PROGRESS)
2. High-priority Task B arrives
3. Agent calls `refresh_context(task_id="task_a", format="minimal")` to save state
4. Agent switches to Task B, completes it
5. Agent calls `refresh_context(task_id="task_a", format="standard")` to restore state
6. Agent resumes Task A using continuation prompt

**Expected Outcome**: Agent can context-switch efficiently without losing track of Task A's progress.

### 5.4 Debugging Failed Dispatches

**Scenario**: A dispatch fails repeatedly, and the orchestrator needs to understand why.

**Workflow**:

1. Dispatch fails 3 times with cryptic error
2. Orchestrator calls `refresh_context(task_id="failed_task", format="detailed", include_dispatches=true, dispatch_limit=10)`
3. Orchestrator receives:
   - Full dispatch history showing repeated failures
   - Error messages from each attempt
   - Git status showing conflicting changes
   - Tech debt item documenting a known blocker
4. Orchestrator identifies the root cause (blocker) and creates a new task to fix it

**Expected Outcome**: Orchestrator can diagnose and resolve the failure without manual intervention.

### 5.5 Orchestrator Session Start

**Scenario**: Orchestrator starts a new session and needs to understand the current project state.

**Workflow**:

1. Orchestrator calls `workspace_handshake(project_root="/path/to/project")`
2. Orchestrator calls `refresh_context(format="narrative")` (no specific task)
3. Orchestrator receives:
   - Summary of all active tasks
   - Recent dispatch activity
   - Open tech debt items
   - Git status across the project
4. Orchestrator uses this to decide which tasks to dispatch next

**Expected Outcome**: Orchestrator has a high-level view of the project and can make informed dispatching decisions.

### 5.6 Long-Running Workflow Resumption

**Scenario**: A multi-phase workflow (e.g., orchestrator) is interrupted mid-execution.

**Workflow**:

1. Orchestrator is in "Phase 3: Implementation" of a 5-phase workflow
2. Session times out
3. New session starts
4. Orchestrator calls `refresh_context(format="detailed", include_task_artifacts=true)`
5. Orchestrator receives:
   - Workflow state (current phase)
   - Completed phases (Phase 1, 2 done)
   - Pending phases (Phase 4, 5 remaining)
   - Artifacts from completed phases
6. Orchestrator resumes at Phase 3

**Expected Outcome**: Orchestrator continues the workflow without restarting from Phase 1.

---

## 6. Implementation Considerations

### 6.1 Performance Optimization

**Challenge**: Aggregating data from multiple sources can be slow.

**Solutions**:

1. **Lazy Loading**: Only load detailed data (git diffs, task artifacts) in "detailed" format
2. **Caching**: Cache git status and recent commits for 60 seconds (most sessions won't need real-time git data)
3. **Parallel Queries**: Query database tables concurrently (tasks, dispatches, tech debt)
4. **Pagination**: Limit dispatch history, commits, and tech debt items to configurable maximums

### 6.2 Token Budget Management

**Challenge**: Context summaries can consume significant tokens.

**Solutions**:

1. **Format Selection**: Default to "standard" format, use "minimal" for frequent polling
2. **Truncation**: Truncate long descriptions, file paths, and error messages
3. **Summarization**: Summarize git diffs (e.g., "+120 -30 lines in 5 files") instead of full diffs
4. **Selective Inclusion**: Only include tech debt items related to the current task

### 6.3 Stale Context Detection

**Challenge**: Context can become stale if the workspace changes externally (e.g., manual git commits).

**Solutions**:

1. **Timestamp Tracking**: Include `generated_at` timestamp in output
2. **Change Detection**: Compare current git HEAD with cached HEAD to detect external changes
3. **Freshness Warnings**: Warn if context is > 1 hour old and git state has changed
4. **Auto-Refresh**: Optionally auto-refresh context if staleness is detected

### 6.4 Security and Privacy

**Challenge**: Context summaries may contain sensitive information (API keys, credentials).

**Solutions**:

1. **Secret Filtering**: Scan git diffs and file contents for common secret patterns (API keys, passwords)
2. **Redaction**: Redact detected secrets in continuation prompts
3. **Opt-Out**: Allow users to disable git diff inclusion via parameter
4. **Audit Logging**: Log when `refresh_context` is called and what data was included

### 6.5 Error Handling

**Challenge**: Context assembly can fail if sources are unavailable (e.g., git not initialized).

**Solutions**:

1. **Graceful Degradation**: If git fails, continue with database-only context
2. **Partial Success**: Return partial context with warnings about missing sources
3. **Error Details**: Include error messages in metadata (e.g., "Git status failed: not a git repository")
4. **Fallback Format**: If "detailed" format fails, fall back to "standard"

---

## 7. Integration with Existing Infrastructure

### 7.1 Database Schema Extensions

**No schema changes required**. The `refresh_context` tool can be implemented using existing tables:

- `tasks`
- `dispatches`
- `tech_debt`
- `research`
- `recon`

**Optional Enhancement**: Add a `sessions` table to track session metadata:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    agent_mode TEXT,
    model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    token_usage INTEGER,
    last_refresh_at TEXT
);
```

### 7.2 MCP Tool Implementation

**Location**: `agent-infra-mcp/src/server_tools/context.rs` (new module)

**Dependencies**:

- `db::AgentDb` (for database queries)
- `std::process::Command` (for git commands)
- `std::fs` (for reading task artifacts)

**Integration**:

- Add `context_router()` to `AgentInfraServer::new()`
- Implement `#[tool_router]` pattern for `refresh_context` tool

**Estimated LOC**: ~400-600 lines (including all format variants)

### 7.3 Orchestrator Integration

**Use Case**: Orchestrator calls `refresh_context` at session start to understand project state.

**Workflow Update** (`.agent/workflows/orchestrator.md`):

```markdown
## Phase 0: Session Initialization (NEW)

1. Call `workspace_handshake(project_root="...")`
2. Call `refresh_context(format="narrative")`
3. Review active tasks, recent dispatches, and tech debt
4. Decide which tasks to prioritize for this session
5. Proceed to Phase 1: Task Selection
```

### 7.4 Skill Integration

**New Skill**: `session-recovery` (`.agent/skills/session-recovery/SKILL.md`)

**Description**: Guides agents on how to use `refresh_context` for session recovery.

**Content**:

- When to call `refresh_context` (timeout, handoff, task switch)
- How to interpret the continuation prompt
- How to verify context freshness
- How to handle stale or incomplete context

---

## 8. Future Enhancements

### 8.1 Intelligent Context Summarization

**Idea**: Use an LLM to generate natural language summaries of complex context (e.g., git diffs, tech debt).

**Benefits**:

- More human-readable summaries
- Better token efficiency (LLM can compress verbose data)
- Adaptive summaries based on task type

**Implementation**:

- Add optional `summarize=true` parameter
- Call Gemini Flash API to summarize verbose sections
- Cache summaries to avoid repeated API calls

### 8.2 Context Diff (Delta Updates)

**Idea**: Instead of regenerating full context, return only what changed since last refresh.

**Benefits**:

- Extreme token efficiency for frequent polling
- Faster execution (no need to re-query unchanged data)

**Implementation**:

- Track last refresh timestamp per session
- Query only changed data (new dispatches, new commits, updated tasks)
- Return delta object: `{ added: [...], updated: [...], removed: [...] }`

### 8.3 Proactive Context Refresh

**Idea**: Automatically refresh context when staleness is detected (e.g., git HEAD changed).

**Benefits**:

- Agents always have fresh context
- Prevents errors due to stale assumptions

**Implementation**:

- Add background watcher that monitors git HEAD, task status
- Trigger `refresh_context` automatically when changes detected
- Notify agent via event or callback

### 8.4 Context Persistence

**Idea**: Save context snapshots to disk for offline review or debugging.

**Benefits**:

- Humans can review what the agent "knew" at any point
- Useful for debugging failed sessions

**Implementation**:

- Add `save_to_file=true` parameter
- Save context JSON to `.agent/context_snapshots/{timestamp}.json`
- Provide tool to load and compare snapshots

### 8.5 Multi-Task Context

**Idea**: Support refreshing context for multiple tasks simultaneously (e.g., all P1 tasks).

**Benefits**:

- Orchestrator can get overview of all high-priority work
- Enables batch decision-making

**Implementation**:

- Add `task_ids: Vec<String>` parameter (mutually exclusive with `task_id`)
- Return array of context objects, one per task
- Aggregate summary showing cross-task dependencies

---

## 9. Success Metrics

### 9.1 Quantitative Metrics

1. **Context Assembly Time**: < 2 seconds for "standard" format, < 5 seconds for "detailed"
2. **Token Efficiency**:
   - Minimal: < 1000 tokens
   - Standard: < 4000 tokens
   - Detailed: < 10000 tokens
3. **Cache Hit Rate**: > 80% for git status queries (within 60-second window)
4. **Error Rate**: < 5% (graceful degradation for missing sources)

### 9.2 Qualitative Metrics

1. **Session Recovery Success**: Agent successfully resumes work after timeout without re-querying sources
2. **Agent Handoff Quality**: Receiving agent understands previous agent's work without confusion
3. **Task Switch Efficiency**: Agent can switch tasks and return without losing context
4. **Orchestrator Effectiveness**: Orchestrator makes informed decisions based on context summary

### 9.3 User Feedback

1. **Agent Satisfaction**: Agents report that continuation prompts are clear and actionable
2. **Human Satisfaction**: Humans reviewing narrative summaries find them accurate and useful
3. **Debugging Utility**: Developers use context snapshots to diagnose failed sessions

---

## 10. Comparison with `prep_orchestrator`

The existing `prep_orchestrator` tool provides similar functionality. Here's how `refresh_context` differs:

| Feature | `prep_orchestrator` | `refresh_context` |
|---------|---------------------|-------------------|
| **Purpose** | Prepare context for orchestrator session start | Resume work after interruption |
| **Scope** | All active tasks, recent dispatches, tech debt | Single task (or all tasks) |
| **Output** | JSON object with aggregated data | JSON + continuation prompt |
| **Verbosity** | Fixed (comprehensive) | Configurable (minimal/standard/detailed/narrative) |
| **Git Integration** | No | Yes (status, diffs, commits) |
| **Task Artifacts** | No | Yes (README, tracking logs) |
| **Continuation Prompt** | No | Yes (self-contained resume prompt) |
| **Use Case** | Orchestrator initialization | Session recovery, agent handoff, task switching |

**Recommendation**: Keep both tools. `prep_orchestrator` is optimized for orchestrator workflows, while `refresh_context` is optimized for individual agent session recovery. They can share underlying logic (e.g., database queries) but serve different use cases.

---

## 11. Recommended Implementation Plan

### Phase 1: Core Implementation (Week 1)

1. Create `server_tools/context.rs` module
2. Implement `refresh_context` tool with "standard" format only
3. Integrate with existing database queries (tasks, dispatches, tech debt)
4. Add basic git status integration (branch, uncommitted changes)
5. Generate simple continuation prompt (template-based)
6. Add to `AgentInfraServer` router composition
7. Write unit tests for context assembly

### Phase 2: Format Variants (Week 2)

1. Implement "minimal" format (token-optimized)
2. Implement "detailed" format (comprehensive)
3. Implement "narrative" format (human-readable markdown)
4. Add format selection logic and parameter validation
5. Add token estimation to metadata
6. Write tests for each format variant

### Phase 3: Advanced Features (Week 3)

1. Add task artifact loading (README, tracking logs)
2. Add git diff integration (summary and full)
3. Add related research/recon loading
4. Implement adaptive continuation prompts (state-based)
5. Add caching for git queries
6. Add error handling and graceful degradation

### Phase 4: Integration & Testing (Week 4)

1. Update orchestrator workflow to use `refresh_context`
2. Create `session-recovery` skill documentation
3. Add integration tests (full workflow scenarios)
4. Performance testing and optimization
5. Documentation and examples
6. User acceptance testing

---

## 12. References

- **Existing Tools**: `prep_orchestrator`, `task(action: "list")`, `dispatch(action: "status")`, `debt(action: "list")`
- **Database Schema**: `agent-infra-mcp/src/db.rs`
- **MCP Server**: `agent-infra-mcp/src/server.rs`
- **Orchestrator Workflow**: `.agent/workflows/orchestrator.md`
- **Maintenance Prompt Library**: `.agent/research/maintenance_prompt_library.md`
- **Standardized Agent Infrastructure**: Knowledge Item (overview.md)

---

## 13. Open Questions

1. **Session Tracking**: Should we add a `sessions` table to track session metadata, or rely on dispatch history?
2. **Context Caching**: How long should we cache git status? 60 seconds? Configurable?
3. **Secret Detection**: Should we implement secret filtering in `refresh_context`, or assume agents handle this separately?
4. **Multi-Task Support**: Should the initial implementation support multi-task context, or defer to Phase 2?
5. **Continuation Prompt Format**: Should continuation prompts be markdown or plain text? Structured (YAML frontmatter) or freeform?

---

**End of Research Report**
