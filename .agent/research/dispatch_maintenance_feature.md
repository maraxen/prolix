# Research: dispatch_maintenance Feature

**Date**: 2026-01-21  
**Researcher**: Antigravity Agent  
**Task ID**: 260120204822  
**Dispatch ID**: d260120210504

---

## Executive Summary

This research proposes a `dispatch_maintenance` MCP tool that automates the batch creation of Jules remote tasks for recurring maintenance work. The feature bridges the gap between the **Maintenance Prompt Library** (47+ discrete prompts) and **Jules CLI** (remote task execution), enabling fully automated "fire and forget" maintenance workflows.

**Key Design Principles**:

1. **Batch Task Creation**: Generate multiple Jules tasks from a single MCP call
2. **Category-Based Selection**: Select prompts by category (dependencies, linting, security, etc.)
3. **Automatic Task Tracking**: Create corresponding database tasks and dispatches
4. **Dry-Run Support**: Preview tasks before creation
5. **Template Substitution**: Inject project-specific context into prompt templates
6. **Error Handling**: Graceful failure with partial success reporting

---

## 1. Jules CLI Integration

### 1.1 Programmatic Task Creation

**Command**: `jules remote new`

**Interface**:

```bash
jules remote new --session "task description" [--repo owner/repo] [--parallel N]
```

**Input Methods**:

1. **Direct String**: `--session "task description"`
2. **Pipe Input**: `echo "task" | jules remote new`
3. **File Input**: `cat task.md | jules remote new`

**Output Format** (stdout):

```
Created remote session for repo: owner/repo
Session ID: abc123def456
Status: queued
```

**Parsing Strategy**:

- Capture stdout to extract Session ID
- Session ID format: alphanumeric string (e.g., `abc123def456`)
- Regex pattern: `Session ID: ([a-zA-Z0-9]+)`

**Error Handling**:

- Exit code 0: Success
- Exit code != 0: Failure (parse stderr for error message)
- Common errors:
  - "No repository found" (need to specify `--repo`)
  - "Authentication failed" (Jules CLI not configured)
  - "Rate limit exceeded" (too many concurrent tasks)

### 1.2 Task Status Checking

**Command**: `jules remote list --session`

**Output Format** (stdout):

```
ID: abc123def456
Repo: owner/repo
Status: queued | in_progress | completed | failed
Created: 2026-01-21 02:30:00
```

**Parsing Strategy**:

- Parse line-by-line output
- Extract ID, Status, Created timestamp
- Map Jules status to internal dispatch status:
  - `queued` → `pending`
  - `in_progress` → `running`
  - `completed` → `completed`
  - `failed` → `failed`

**Polling Strategy**:

- **Not recommended**: Jules tasks are async (hours+), polling is inefficient
- **Alternative**: Store Jules Session ID in dispatch record, check on-demand via `dispatch(action: "status")`

### 1.3 Task Result Retrieval

**Command**: `jules remote pull --session <id>`

**Output Format** (stdout):

```
Session ID: abc123def456
Status: completed
Files changed: 3
Patch available: yes

[Diff output follows...]
```

**Parsing Strategy**:

- Extract status and file count
- Capture diff output for review
- Optionally apply with `--apply` flag (requires manual review first)

**Integration with Dispatch System**:

- When Jules task completes, update dispatch status to `completed`
- Store diff output in dispatch `result` field
- Optionally create a new dispatch to review and apply the patch

### 1.4 Repository Detection

**Challenge**: Jules requires `--repo owner/repo` if not in a git repository.

**Solutions**:

1. **Auto-detect from git remote**:

   ```bash
   git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/'
   ```

   Output: `owner/repo`

2. **Fallback to config**: Store default repo in `.agent/config.toml`:

   ```toml
   [jules]
   default_repo = "owner/repo"
   ```

3. **Require explicit parameter**: Force users to specify repo in `dispatch_maintenance` call

**Recommendation**: Auto-detect from git remote, fallback to config, error if neither available.

### 1.5 Error Handling Patterns

**Scenario 1: Jules CLI Not Installed**

- **Detection**: `jules` command not found
- **Response**: Return error with installation instructions
- **Error Message**: "Jules CLI not found. Install with: npm install -g @jules-cli/cli"

**Scenario 2: Authentication Failure**

- **Detection**: Exit code 1, stderr contains "authentication"
- **Response**: Return error with auth instructions
- **Error Message**: "Jules authentication failed. Run: jules auth login"

**Scenario 3: Rate Limit**

- **Detection**: Exit code 1, stderr contains "rate limit"
- **Response**: Delay and retry, or fail gracefully
- **Error Message**: "Jules rate limit exceeded. Retry in 60 seconds."

**Scenario 4: Partial Batch Failure**

- **Detection**: Some tasks succeed, some fail
- **Response**: Return partial success with details
- **Error Message**: "Created 5/10 tasks. Failed: [list of failed prompts]"

---

## 2. Task Generation Logic

### 2.1 Prompt Category Mapping

The **Maintenance Prompt Library** defines 12 categories with 47+ prompts. Each prompt should map to a Jules task template.

**Category-to-Template Mapping**:

| Category | Prompts | Jules Template Type | Auto-Dispatchable |
|----------|---------|---------------------|-------------------|
| Dependency Management | 6 | Lint/Type Fix | ✅ |
| Code Quality & Linting | 6 | Lint/Type Fix | ✅ |
| Test Health & Coverage | 6 | Unit Test Addition | ⚠️ (context-dependent) |
| Security & Vulnerability | 6 | Lint/Type Fix | ✅ |
| Documentation Maintenance | 5 | Docstring/Documentation | ✅ |
| Build & Compilation Health | 5 | Lint/Type Fix | ✅ |
| Performance & Profiling | 4 | Custom | ⚠️ (manual review) |
| Database & Schema Management | 4 | Custom | ⚠️ (manual review) |
| Infrastructure & Configuration | 4 | Custom | ⚠️ (cloud access) |
| Git & Repository Health | 4 | Custom | ✅ |
| Monitoring & Observability | 3 | Custom | ❌ (log access) |
| Technical Debt Management | 4 | Custom | ⚠️ (context-dependent) |

**Auto-Dispatchable Criteria**:

- ✅ **Fully Auto**: Requires only file system access and command execution
- ⚠️ **Partially Auto**: May need context or manual review
- ❌ **Not Auto**: Requires external service access or manual decisions

### 2.2 Template Substitution

Each prompt template should support variable substitution for project-specific context.

**Common Variables**:

- `{project_root}`: Absolute path to project root
- `{project_name}`: Project name (from git repo or config)
- `{language}`: Primary language (detected from file extensions)
- `{package_manager}`: npm, cargo, pip, etc. (detected from manifest files)
- `{test_framework}`: pytest, vitest, jest, etc. (detected from config)
- `{linter}`: ruff, eslint, clippy, etc. (detected from config)
- `{codestyle_path}`: Path to `.agent/codestyles/{language}.md`

**Example Template** (`dep_audit`):

```
Title: Audit dependencies for security vulnerabilities

## Context
Project: {project_name}
Package Manager: {package_manager}
Manifest: {manifest_path}

## Requirements
- Run `{package_manager} audit` or equivalent security scanner
- Report all vulnerabilities with CVSS scores
- Categorize by severity (critical, high, medium, low)
- Suggest fix versions for each vulnerability

## Acceptance Criteria
- Output saved to `.agent/reports/dep_audit_{timestamp}.json`
- Report includes: package, current version, fixed version, severity, CVSS score
- No critical or high vulnerabilities remain unfixed (or documented as accepted risk)
```

**Substitution Logic**:

1. Detect project context (language, package manager, etc.)
2. Replace `{variable}` placeholders with actual values
3. Validate that all required variables are available
4. Error if critical variables are missing (e.g., `{package_manager}` for `dep_audit`)

### 2.3 Scope Filtering

**Problem**: Not all prompts are relevant to all projects (e.g., `db_migration_verify` for projects without databases).

**Solution**: Support scope filtering based on project characteristics.

**Scope Dimensions**:

1. **Language**: python, rust, typescript, etc.
2. **Framework**: django, flask, express, next.js, etc.
3. **Components**: database, frontend, backend, cli, etc.
4. **Tooling**: docker, kubernetes, terraform, etc.

**Scope Detection**:

- **Language**: File extension analysis (`.py`, `.rs`, `.ts`)
- **Framework**: Config file analysis (`package.json`, `Cargo.toml`, `pyproject.toml`)
- **Components**: Directory structure analysis (`/db`, `/frontend`, `/backend`)
- **Tooling**: Config file presence (`Dockerfile`, `terraform/`, `.github/workflows/`)

**Filtering Logic**:

```rust
fn filter_prompts_by_scope(prompts: Vec<Prompt>, scope: ProjectScope) -> Vec<Prompt> {
    prompts.into_iter().filter(|p| {
        // Only include prompts relevant to this project
        match p.category {
            "database" => scope.has_database,
            "infrastructure" => scope.has_iac,
            "frontend" => scope.has_frontend,
            _ => true, // Generic prompts always included
        }
    }).collect()
}
```

### 2.4 Dry-Run Mode

**Purpose**: Preview tasks before creating them in Jules.

**Behavior**:

- Generate all task descriptions with substituted variables
- Return list of tasks that would be created
- Do NOT call `jules remote new`
- Do NOT create database records

**Output Format**:

```json
{
  "dry_run": true,
  "tasks_to_create": [
    {
      "category": "dependency_management",
      "prompt_name": "dep_audit",
      "jules_task": "Title: Audit dependencies...\n\n## Context\n...",
      "estimated_duration": "2m"
    },
    {
      "category": "code_quality",
      "prompt_name": "lint_check",
      "jules_task": "Title: Check linting...\n\n## Context\n...",
      "estimated_duration": "1m"
    }
  ],
  "total_tasks": 2,
  "estimated_total_duration": "3m"
}
```

**Use Case**: Review tasks before batch creation, adjust scope or categories if needed.

---

## 3. Database Schema Extension

### 3.1 Current Schema Review

**Existing Tables**:

- `tasks`: Core task tracking
- `dispatches`: Dispatch execution tracking
- `tech_debt`: Technical debt items
- `prompts`: Reusable prompt templates
- `research`, `recon`: Structured findings

**Relevant Fields in `dispatches`**:

- `id`: Dispatch ID (e.g., `d260120210504`)
- `target`: Dispatch target (e.g., `jules`, `cli`, `antigravity`)
- `task_id`: Associated task ID (optional)
- `model`: Model used (optional, not applicable for Jules)
- `mode`: Agent mode (optional)
- `prompt_full`: Full prompt text
- `status`: `pending`, `running`, `completed`, `failed`
- `result`: Execution result summary

### 3.2 Schema Extensions for Jules Integration

**Option 1: Add Jules-Specific Fields to `dispatches`**

```sql
ALTER TABLE dispatches ADD COLUMN jules_session_id TEXT;
ALTER TABLE dispatches ADD COLUMN jules_repo TEXT;
ALTER TABLE dispatches ADD COLUMN maintenance_category TEXT;
ALTER TABLE dispatches ADD COLUMN maintenance_batch_id TEXT;
```

**Fields**:

- `jules_session_id`: Jules Session ID (e.g., `abc123def456`)
- `jules_repo`: Repository (e.g., `owner/repo`)
- `maintenance_category`: Prompt category (e.g., `dependency_management`)
- `maintenance_batch_id`: Batch ID for grouping (e.g., `batch_260121_023000`)

**Pros**:

- Simple, no new tables
- All dispatch data in one place
- Easy to query Jules tasks: `SELECT * FROM dispatches WHERE jules_session_id IS NOT NULL`

**Cons**:

- Adds Jules-specific fields to generic dispatch table
- May clutter schema if other dispatch types are added

**Option 2: Create Separate `maintenance_dispatches` Table**

```sql
CREATE TABLE maintenance_dispatches (
    id TEXT PRIMARY KEY,
    dispatch_id TEXT NOT NULL,
    jules_session_id TEXT,
    jules_repo TEXT,
    category TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    batch_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (dispatch_id) REFERENCES dispatches(id) ON DELETE CASCADE
);
```

**Pros**:

- Clean separation of concerns
- Easier to add maintenance-specific fields later
- Doesn't clutter `dispatches` table

**Cons**:

- Requires JOIN to get full dispatch + maintenance data
- More complex queries

**Recommendation**: **Option 1** (add fields to `dispatches`). Simpler, and the fields are optional (NULL for non-Jules dispatches).

### 3.3 Migration Script

**File**: `.agent/migrations/002_add_jules_fields.sql`

```sql
-- Add Jules-specific fields to dispatches table
ALTER TABLE dispatches ADD COLUMN jules_session_id TEXT;
ALTER TABLE dispatches ADD COLUMN jules_repo TEXT;
ALTER TABLE dispatches ADD COLUMN maintenance_category TEXT;
ALTER TABLE dispatches ADD COLUMN maintenance_batch_id TEXT;

-- Create index for efficient Jules task queries
CREATE INDEX idx_dispatches_jules_session ON dispatches(jules_session_id);
CREATE INDEX idx_dispatches_maintenance_batch ON dispatches(maintenance_batch_id);
```

**Migration Logic** (in `db.rs`):

```rust
pub fn migrate_002_add_jules_fields(&self) -> Result<()> {
    let migration = fs::read_to_string(".agent/migrations/002_add_jules_fields.sql")?;
    self.conn.execute_batch(&migration)?;
    Ok(())
}
```

### 3.4 Updated `DbDispatch` Struct

```rust
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct DbDispatch {
    pub id: String,
    pub target: String,
    pub task_id: Option<String>,
    pub model: Option<String>,
    pub mode: Option<String>,
    pub depends_on: Option<String>,
    pub prompt_hash: String,
    pub prompt_preview: String,
    pub prompt_full: String,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub claimed_by: Option<String>,
    pub claimed_at: Option<String>,
    pub result: Option<String>,
    
    // NEW: Jules-specific fields
    pub jules_session_id: Option<String>,
    pub jules_repo: Option<String>,
    pub maintenance_category: Option<String>,
    pub maintenance_batch_id: Option<String>,
}
```

---

## 4. MCP Tool Interface

### 4.1 Tool Definition

**Tool Name**: `dispatch_maintenance`

**Description**: Batch create Jules remote tasks for recurring maintenance work based on prompt categories.

**Category**: Orchestration / Maintenance

### 4.2 Input Parameters

```rust
#[derive(Debug, serde::Deserialize)]
pub struct DispatchMaintenanceRequest {
    /// Prompt categories to include (e.g., ["dependency_management", "code_quality"])
    /// If empty, includes all auto-dispatchable categories
    pub categories: Option<Vec<String>>,
    
    /// Specific prompt names to include (e.g., ["dep_audit", "lint_check"])
    /// Overrides category selection if provided
    pub prompts: Option<Vec<String>>,
    
    /// Project scope filter (e.g., {"language": "rust", "has_database": true})
    /// Auto-detected if not provided
    pub scope: Option<serde_json::Value>,
    
    /// Repository for Jules tasks (e.g., "owner/repo")
    /// Auto-detected from git remote if not provided
    pub repo: Option<String>,
    
    /// Dry-run mode: preview tasks without creating them
    /// Default: false
    pub dry_run: Option<bool>,
    
    /// Create corresponding task records in the database
    /// Default: true
    pub create_tasks: Option<bool>,
    
    /// Batch ID for grouping (auto-generated if not provided)
    pub batch_id: Option<String>,
    
    /// Parallel execution count for each Jules task
    /// Default: 1
    pub parallel: Option<u8>,
    
    /// Auto-apply patches when Jules tasks complete
    /// Default: false (requires manual review)
    pub auto_apply: Option<bool>,
}
```

### 4.3 Output Format

**Success Response** (JSON):

```json
{
  "batch_id": "batch_260121_023000",
  "dry_run": false,
  "tasks_created": [
    {
      "dispatch_id": "d260121023001",
      "task_id": "260121023001",
      "jules_session_id": "abc123def456",
      "category": "dependency_management",
      "prompt_name": "dep_audit",
      "status": "pending",
      "repo": "owner/repo"
    },
    {
      "dispatch_id": "d260121023002",
      "task_id": "260121023002",
      "jules_session_id": "def456ghi789",
      "category": "code_quality",
      "prompt_name": "lint_check",
      "status": "pending",
      "repo": "owner/repo"
    }
  ],
  "tasks_failed": [
    {
      "category": "security",
      "prompt_name": "secrets_rotate",
      "error": "Prompt not auto-dispatchable (requires manual intervention)"
    }
  ],
  "summary": {
    "total_requested": 10,
    "created": 8,
    "failed": 2,
    "skipped": 0
  }
}
```

**Error Response**:

```json
{
  "error": "Jules CLI not found. Install with: npm install -g @jules-cli/cli",
  "code": "JULES_NOT_INSTALLED"
}
```

### 4.4 Related Tools

**Upstream Tools** (provide data to dispatch_maintenance):

- `workspace_handshake`: Establishes active workspace
- `prompt(action: "list")`: Lists available maintenance prompts
- `skill(action: "load")`: Loads Jules remote skill for task templates

**Downstream Tools** (consume dispatch_maintenance output):

- `dispatch(action: "status")`: Check status of Jules tasks
- `task(action: "list")`: List created maintenance tasks
- `dispatch(action: "complete")`: Mark Jules tasks as completed when pulled

**Composition Pattern**:

```
workspace_handshake() 
  → dispatch_maintenance(categories=["dependency_management", "code_quality"], dry_run=true)
  → [review output]
  → dispatch_maintenance(categories=["dependency_management", "code_quality"], dry_run=false)
  → dispatch(action: "status")(status="pending", limit=20)
```

---

## 5. Orchestration Patterns

### 5.1 Weekly Maintenance Workflow

**Trigger**: Cron job every Sunday at midnight

**Workflow**:

```yaml
name: weekly_maintenance
schedule: "0 0 * * 0"  # Sunday midnight
steps:
  - workspace_handshake(project_root="/path/to/project")
  - dispatch_maintenance(
      categories=[
        "dependency_management",
        "code_quality",
        "security",
        "git"
      ],
      dry_run=false,
      create_tasks=true,
      batch_id="weekly_${date}"
    )
  - # Wait for Jules tasks to complete (hours+)
  - # Manual review and apply patches
```

**Expected Output**: 10-15 Jules tasks created, queued for execution.

### 5.2 Pre-Release Checklist

**Trigger**: Manual, before major release

**Workflow**:

```yaml
name: pre_release_checklist
trigger: manual
steps:
  - workspace_handshake(project_root="/path/to/project")
  - dispatch_maintenance(
      prompts=[
        "test_run_all",
        "coverage_report",
        "security_scan_deps",
        "docs_changelog",
        "docs_examples_verify",
        "build_size_check",
        "sbom_generate"
      ],
      dry_run=false,
      create_tasks=true,
      batch_id="pre_release_${version}"
    )
  - # Wait for all tasks to complete
  - # Review results, block release if critical issues found
```

**Expected Output**: 7 Jules tasks created, all must pass before release.

### 5.3 On-Demand Maintenance

**Trigger**: Manual, when developer notices issues

**Workflow**:

```bash
# Developer notices lint errors
dispatch_maintenance(
  prompts=["lint_check", "lint_fix_auto"],
  dry_run=false
)

# Developer wants to update dependencies
dispatch_maintenance(
  categories=["dependency_management"],
  dry_run=false
)
```

**Expected Output**: 1-6 Jules tasks created, depending on selection.

### 5.4 Continuous Maintenance (Daily)

**Trigger**: Cron job every day at 2 AM

**Workflow**:

```yaml
name: daily_maintenance
schedule: "0 2 * * *"  # Daily at 2 AM
steps:
  - workspace_handshake(project_root="/path/to/project")
  - dispatch_maintenance(
      prompts=[
        "security_scan_deps",
        "db_backup_verify",
        "logs_error_analysis"
      ],
      dry_run=false,
      create_tasks=true,
      batch_id="daily_${date}"
    )
```

**Expected Output**: 3 Jules tasks created daily.

### 5.5 Conditional Dispatch (Event-Driven)

**Trigger**: On specific events (e.g., dependency update, security alert)

**Workflow**:

```python
# Pseudo-code for event-driven dispatch
def on_dependency_update(package, version):
    dispatch_maintenance(
        prompts=["dep_audit", "test_run_all"],
        dry_run=False,
        batch_id=f"dep_update_{package}_{version}"
    )

def on_security_alert(cve_id):
    dispatch_maintenance(
        prompts=["security_scan_deps", "dep_upgrade_patch"],
        dry_run=False,
        batch_id=f"security_alert_{cve_id}"
    )
```

**Expected Output**: 2 Jules tasks created per event.

---

## 6. Implementation Strategy

### 6.1 Phase 1: Core Infrastructure (Week 1)

**Goals**:

1. Implement Jules CLI wrapper functions (create, list, pull)
2. Add Jules-specific fields to `dispatches` table (migration 002)
3. Implement basic `dispatch_maintenance` tool (single prompt, no batch)
4. Test end-to-end: create Jules task, track in database, check status

**Deliverables**:

- `src/jules.rs`: Jules CLI wrapper module
- `.agent/migrations/002_add_jules_fields.sql`: Schema migration
- `src/server_tools/maintenance.rs`: `dispatch_maintenance` tool (basic)
- Unit tests for Jules CLI wrapper
- Integration test: create + track Jules task

### 6.2 Phase 2: Batch & Template System (Week 2)

**Goals**:

1. Implement category-based prompt selection
2. Implement template substitution (variable replacement)
3. Implement batch task creation (multiple prompts in one call)
4. Implement dry-run mode
5. Add scope filtering (auto-detect project characteristics)

**Deliverables**:

- `src/templates.rs`: Template substitution engine
- `src/scope.rs`: Project scope detection
- Updated `dispatch_maintenance` tool (batch support)
- Tests for template substitution and scope filtering

### 6.3 Phase 3: Orchestration & Automation (Week 3)

**Goals**:

1. Create maintenance workflow definitions (weekly, daily, pre-release)
2. Implement batch status checking (poll all Jules tasks in a batch)
3. Implement auto-apply logic (optional, with safety checks)
4. Add error handling and partial failure reporting
5. Create orchestrator integration (call `dispatch_maintenance` from workflows)

**Deliverables**:

- `.agent/workflows/weekly_maintenance.md`: Weekly maintenance workflow
- `.agent/workflows/pre_release_checklist.md`: Pre-release workflow
- `src/server_tools/maintenance.rs`: Auto-apply logic
- Integration tests for workflows

### 6.4 Phase 4: Monitoring & Reporting (Week 4)

**Goals**:

1. Implement batch result aggregation (collect all Jules task results)
2. Generate maintenance reports (summary of issues found, fixes applied)
3. Add alerting for critical issues (e.g., security vulnerabilities)
4. Create dashboard or summary view (all maintenance batches, status)
5. Documentation and user guides

**Deliverables**:

- `src/server_tools/reports.rs`: Batch result aggregation
- `.agent/reports/`: Maintenance report templates
- Documentation: `docs/dispatch_maintenance.md`
- User guide: `docs/maintenance_workflows.md`

---

## 7. Error Handling & Edge Cases

### 7.1 Jules CLI Errors

**Error**: Jules CLI not installed

- **Detection**: `jules` command not found
- **Response**: Return error with installation instructions
- **Recovery**: None (user must install Jules)

**Error**: Authentication failure

- **Detection**: Exit code 1, stderr contains "authentication"
- **Response**: Return error with auth instructions
- **Recovery**: None (user must authenticate)

**Error**: Rate limit exceeded

- **Detection**: Exit code 1, stderr contains "rate limit"
- **Response**: Delay and retry, or fail gracefully
- **Recovery**: Retry after 60 seconds (configurable)

### 7.2 Template Substitution Errors

**Error**: Missing required variable

- **Detection**: Template contains `{variable}` but variable not available
- **Response**: Skip prompt, log error
- **Recovery**: Continue with other prompts (partial success)

**Error**: Invalid variable value

- **Detection**: Variable value doesn't match expected format (e.g., invalid path)
- **Response**: Skip prompt, log error
- **Recovery**: Continue with other prompts (partial success)

### 7.3 Batch Creation Errors

**Error**: Partial batch failure (some tasks succeed, some fail)

- **Detection**: Some `jules remote new` calls succeed, some fail
- **Response**: Return partial success with details
- **Recovery**: User can retry failed prompts individually

**Error**: All tasks fail

- **Detection**: All `jules remote new` calls fail
- **Response**: Return error with details
- **Recovery**: User must fix underlying issue (e.g., auth, rate limit)

### 7.4 Database Errors

**Error**: Failed to insert dispatch record

- **Detection**: Database insert fails (e.g., constraint violation)
- **Response**: Rollback Jules task creation (if possible), return error
- **Recovery**: None (critical error, requires manual intervention)

**Error**: Failed to update dispatch status

- **Detection**: Database update fails
- **Response**: Log error, continue (Jules task still exists)
- **Recovery**: Manual database update

---

## 8. Security & Safety Considerations

### 8.1 Credential Management

**Risk**: Jules CLI requires authentication, credentials may be exposed.

**Mitigation**:

- Jules CLI stores credentials securely (OAuth tokens)
- MCP tool does NOT handle credentials directly
- Assume Jules CLI is already authenticated (prerequisite)

### 8.2 Auto-Apply Safety

**Risk**: Auto-applying Jules patches could introduce bugs or break code.

**Mitigation**:

- Default `auto_apply=false` (requires manual review)
- If `auto_apply=true`, run tests before applying
- Create backup before applying (git stash or commit)
- Rollback on test failure

**Safety Checklist** (for auto-apply):

1. Jules task status is `completed`
2. Patch applies cleanly (no conflicts)
3. All tests pass after applying
4. No critical lint errors introduced
5. Git working directory is clean (or changes are stashed)

### 8.3 Prompt Injection

**Risk**: User-provided variables could inject malicious commands into Jules tasks.

**Mitigation**:

- Sanitize all user-provided variables (escape special characters)
- Validate variable values against expected formats (e.g., paths, package names)
- Use allowlist for categories and prompt names (no arbitrary prompts)

### 8.4 Resource Limits

**Risk**: Creating too many Jules tasks could overwhelm the system or hit rate limits.

**Mitigation**:

- Limit batch size (max 20 tasks per call, configurable)
- Respect Jules rate limits (delay between task creations)
- Implement exponential backoff on rate limit errors

---

## 9. Success Metrics

### 9.1 Quantitative Metrics

1. **Batch Creation Time**: < 5 seconds for 10 tasks
2. **Success Rate**: > 90% of tasks created successfully
3. **Template Substitution Accuracy**: 100% (no missing variables for auto-dispatchable prompts)
4. **Jules Task Completion Rate**: > 80% (within 24 hours)

### 9.2 Qualitative Metrics

1. **Developer Satisfaction**: Developers find batch maintenance useful and time-saving
2. **Code Quality Improvement**: Fewer lint errors, security vulnerabilities, and outdated dependencies
3. **Automation Coverage**: > 50% of maintenance tasks automated via `dispatch_maintenance`

### 9.3 Operational Metrics

1. **Weekly Maintenance Execution**: Runs successfully every Sunday
2. **Pre-Release Checklist Compliance**: 100% of releases run pre-release checklist
3. **Issue Detection Rate**: > 10 issues found per week via automated maintenance

---

## 10. Comparison with Manual Jules Dispatch

| Feature | Manual Jules Dispatch | `dispatch_maintenance` |
|---------|----------------------|------------------------|
| **Task Creation** | One at a time | Batch (10-20 tasks) |
| **Template Substitution** | Manual | Automatic |
| **Database Tracking** | Manual (add to matrix) | Automatic |
| **Scope Filtering** | Manual (choose relevant prompts) | Automatic (project detection) |
| **Dry-Run** | No | Yes |
| **Batch Grouping** | No | Yes (batch_id) |
| **Error Handling** | Manual | Automatic (partial success) |
| **Time to Create 10 Tasks** | ~10 minutes | ~5 seconds |

**Recommendation**: Use `dispatch_maintenance` for recurring maintenance, manual dispatch for one-off tasks.

---

## 11. Future Enhancements

### 11.1 Intelligent Prompt Selection

**Idea**: Use LLM to analyze project and suggest relevant maintenance prompts.

**Benefits**:

- Better scope filtering (beyond simple file detection)
- Adaptive to project-specific needs
- Discovers maintenance gaps

**Implementation**:

- Analyze codebase structure, dependencies, and recent commits
- Call Gemini Flash API to suggest prompts
- Return ranked list of prompts with rationale

### 11.2 Scheduled Maintenance

**Idea**: Built-in cron-like scheduler for recurring maintenance.

**Benefits**:

- No external cron job needed
- Integrated with MCP system
- Configurable schedules per project

**Implementation**:

- Add `schedules` table to database
- Background worker polls schedules and triggers `dispatch_maintenance`
- Web UI or CLI to manage schedules

### 11.3 Maintenance Dashboard

**Idea**: Web dashboard showing all maintenance batches, status, and results.

**Benefits**:

- Visual overview of maintenance health
- Easy to spot failures or blockers
- Historical trends (issues found over time)

**Implementation**:

- Web server serving dashboard (React or similar)
- Query database for batch status and results
- Charts and graphs for trends

### 11.4 Adaptive Batch Sizing

**Idea**: Automatically adjust batch size based on Jules capacity and rate limits.

**Benefits**:

- Maximize throughput without hitting rate limits
- Adapt to changing Jules capacity

**Implementation**:

- Track Jules rate limit errors
- Reduce batch size on rate limit, increase on success
- Exponential backoff and recovery

### 11.5 Result Aggregation & Reporting

**Idea**: Aggregate results from all Jules tasks in a batch and generate summary report.

**Benefits**:

- Single report for all maintenance issues
- Prioritized list of issues to fix
- Trend analysis (issues over time)

**Implementation**:

- Poll all Jules tasks in batch until complete
- Extract issues from each task result
- Generate markdown report with categorized issues
- Store in `.agent/reports/maintenance_${batch_id}.md`

---

## 12. References

- **Maintenance Prompt Library**: `.agent/research/maintenance_prompt_library.md`
- **Jules Remote Skill**: `.agent/skills/jules-remote/SKILL.md`
- **Database Schema**: `agent-infra-mcp/src/db.rs`
- **Dispatch System**: `agent-infra-mcp/src/server_tools/dispatch.rs`
- **Jules CLI Documentation**: <https://github.com/jiahao42/jules-cli>

---

## 13. Open Questions

1. **Jules API Stability**: Is the Jules CLI output format stable? Should we use JSON output if available?
2. **Batch Size Limit**: What's the optimal batch size? 10? 20? 50?
3. **Auto-Apply Default**: Should `auto_apply` ever be true by default, or always require explicit opt-in?
4. **Prompt Storage**: Should maintenance prompts be stored in the database (`prompts` table) or as markdown files?
5. **Scheduling**: Should scheduling be built into `dispatch_maintenance`, or handled externally (cron, GitHub Actions)?
6. **Result Format**: What format should Jules task results be stored in? Plain text diff? Structured JSON?

---

**End of Research Report**
