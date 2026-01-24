---
description: Complete orchestrator session workflow with phased dispatch, review gates, and parallel thread management
---

# Orchestrator Workflow

The orchestrator operates in **phased loops** with explicit review gates. Each phase dispatches work, reviews results, and decides next steps.

## Prerequisites

- MCP server running (`agent-infra-mcp`)
- SQLite database exists (`.agent/agent.db`)
- Load skills: `orchestration`, `gemini-cli-headless`

---

## The Orchestration Loop

```
┌──────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LOOP                         │
├──────────────────────────────────────────────────────────────┤
│  1. RECON PHASE                                              │
│     ├── Dispatch @recon agents (parallel if multiple targets)│
│     ├── Track threads in DB                                  │
│     └── REVIEW: Synthesize findings                          │
│                                                              │
│  2. PLANNING PHASE                                           │
│     ├── Dispatch @oracle/@architect for strategy             │
│     ├── Create implementation plan                           │
│     └── REVIEW: Validate plan completeness                   │
│                                                              │
│  3. USER CONSULTATION (MANDATORY)                            │
│     ├── Present: Findings + Plan + Risks                     │
│     ├── Get: Approval / Modifications / Abort                │
│     └── GATE: Do NOT proceed without explicit approval       │
│                                                              │
│  4. EXECUTION PHASE                                          │
│     ├── Dispatch @fixer/@flash (parallel when independent)   │
│     ├── Track threads in DB                                  │
│     └── REVIEW: Verify each completion                       │
│                                                              │
│  5. VERIFICATION PHASE                                       │
│     ├── Run tests / visual checks                            │
│     ├── Dispatch @multimodal-looker if UI                    │
│     └── REVIEW: Confirm success criteria met                 │
│                                                              │
│  6. COMPLETION                                               │
│     ├── Update task status                                   │
│     ├── Record findings/lessons                              │
│     └── Archive if done                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Recon

### Purpose

Gather information before planning. Never plan without recon.

### Dispatch Format

```markdown
# Recon Dispatch: {target}

## Agent Mode
@recon (load: .agent/agents/recon.md)

## Skills Context
- orchestration: Dispatch patterns, output specification
- {domain_skill}: Domain-specific guidance

## Task Context
- Matrix ID: {id}
- Thread ID: {thread_id} (for parallel tracking)
- Output: .agent/tasks/{id}_{name}/artifacts/recon_{target}.md

## Objective
{what to investigate}

## Success Criteria
- [ ] {specific finding 1}
- [ ] {specific finding 2}

## Output Format
```json
{
  "target": "{target}",
  "status": "found|not_found|partial",
  "summary": "one-line",
  "findings": { ... },
  "recommendations": [...]
}
```

```

### Parallel Recon Threads

When multiple recon targets exist, dispatch in parallel and track:

```

# Record each dispatch

dispatch(target: "cli", prompt: "...", task_id: "{id}")

# Track thread

INSERT INTO dispatches with unique thread_id

# Query active threads

dispatch(action: "status")(task_id: "{id}")

```

### Review Gate

After all recon threads complete:

1. Collect all recon artifacts
2. Synthesize into unified findings
3. Identify gaps requiring more recon
4. Decide: More recon needed? → Loop. Ready to plan? → Phase 2.

---

## Phase 2: Planning

### Purpose
Create actionable implementation plan based on recon findings.

### Dispatch Format

```markdown
# Planning Dispatch: {task_description}

## Agent Mode
@oracle (load: .agent/agents/oracle.md)

## Skills Context
- orchestration: Phased execution patterns
- planning-with-files: File-based planning methodology
- {architecture_skill}: If architectural decisions needed

## Research Context
Recon findings: .agent/tasks/{id}_{name}/artifacts/recon_*.md

## Task Context
- Matrix ID: {id}
- Output: .agent/tasks/{id}_{name}/artifacts/implementation_plan.md

## Objective
Create implementation plan for: {description}

## Plan Requirements
- Phases with dependencies
- Files to modify per phase
- Risk assessment
- Rollback strategy
- Success criteria per phase

## Output Format
Markdown with phases, each containing:
- Objective
- Files
- Steps
- Verification
```

### Review Gate

After planning:

1. Review plan completeness
2. Identify missing details
3. Assess risk level
4. **MANDATORY: Proceed to User Consultation**

---

## Phase 3: User Consultation (MANDATORY GATE)

### Purpose

Get explicit user approval before execution. **Never skip this phase.**

### Presentation Format

```markdown
## Orchestrator Report: {task_description}

### Recon Summary
{synthesized findings}

### Proposed Plan
{plan overview with phases}

### Risk Assessment
- **High Risk**: {list items requiring caution}
- **Medium Risk**: {list items to monitor}
- **Low Risk**: {routine items}

### Estimated Effort
- Phases: {N}
- Parallel threads: {M}
- Time estimate: {range}

### Decision Required
1. ✅ **Approve**: Proceed with execution
2. ✏️ **Modify**: Adjust plan (specify changes)
3. 🔄 **More Recon**: Need additional information
4. 🛑 **Abort**: Cancel task

**Awaiting your decision.**
```

### Gate Rules

- **No response = No execution**
- **Approval must be explicit** ("proceed", "approved", "go ahead")
- **Modifications require re-planning**
- **Abort updates task status to BLOCKED**

---

## Phase 4: Execution

### Purpose

Execute the approved plan via specialist agents.

### Dispatch Format

```markdown
# Execution Dispatch: Phase {N} - {phase_name}

## Agent Mode
@fixer (load: .agent/agents/fixer.md)

## Skills Context
- {skill_1}: {key instructions from SKILL.md}
- {skill_2}: {key instructions from SKILL.md}

## Task Context
- Matrix ID: {id}
- Thread ID: {thread_id}
- Phase: {N} of {total}
- Dependencies: {completed phases}

## Plan Reference
Implementation plan: .agent/tasks/{id}_{name}/artifacts/implementation_plan.md
This phase: Section {N}

## Files to Modify
- {file_1}
- {file_2}

## Objective
{phase objective}

## Success Criteria
- [ ] {verifiable criterion 1}
- [ ] {verifiable criterion 2}

## Output
- Modified files
- Summary to: .agent/tasks/{id}_{name}/tracking/phase_{N}_complete.md
```

### Parallel Execution

When phases are independent:

```
Phase 1: Completed ✓
├── Phase 2a: Thread A (dispatched)
├── Phase 2b: Thread B (dispatched)
└── Phase 2c: Thread C (dispatched)
Phase 3: Blocked (waiting for 2a, 2b, 2c)
```

Track with:

```
dispatch(action: "status")(task_id: "{id}")
# Returns all threads with status
```

### Review Gate

After each phase/thread:

1. Verify changes made correctly
2. Run relevant tests
3. Check for side effects
4. Phase failed? → Consult user or retry with modifications
5. All phases complete? → Phase 5

---

## Phase 5: Verification

### Purpose

Confirm all success criteria met before marking complete.

### Verification Checklist

```markdown
## Verification: {task_description}

### Automated Checks
- [ ] Tests pass: `npm test` / `cargo test` / `pytest`
- [ ] Build succeeds: `npm run build` / `cargo build`
- [ ] Lint clean: `npm run lint` / `cargo clippy`

### Manual Checks
- [ ] Code review: Changes match plan
- [ ] Functionality: Feature works as expected
- [ ] Side effects: No regressions

### Visual Checks (if UI)
- [ ] Screenshots compared
- [ ] @multimodal-looker verification

### Documentation
- [ ] README updated if needed
- [ ] Comments added for complex logic
```

### Review Gate

- All checks pass? → Phase 6
- Failures? → Return to Phase 4 or consult user

---

## Phase 6: Completion

### MCP Tool Sequence

```
# 1. Update task status
task(action: "update")(id: "{id}", status: "DONE")

# 2. Record findings (if any structured learnings)
report(action: "recon")(target: "...", task_id: "{id}", findings: {...})

# 3. Record tech debt (if discovered)
debt(action: "add")(title: "...", category: "...", description: "...", related_task_id: "{id}")

# 4. Archive (optional, for completed + aged tasks)
task(action: "archive")(id: "{id}", archive_related: true)

# 5. Record orchestration lesson (if significant)
# Add to .agent/ORCHESTRATION.md lessons table
```

---

## Thread Management

### Tracking Active Threads

```sql
-- Query all pending dispatches for a task
SELECT id, target, status, created_at 
FROM dispatches 
WHERE task_id = '{id}' AND status IN ('pending', 'running');
```

Via MCP:

```
dispatch(action: "status")(task_id: "{id}")
```

### Thread States

| State | Meaning |
|-------|---------|
| `pending` | Dispatched, not yet started |
| `running` | Agent working |
| `completed` | Successfully finished |
| `failed` | Error occurred |

### Handling Failed Threads

1. Review failure reason
2. Decide: Retry? Modify? Consult user?
3. If retry: New dispatch with adjusted prompt
4. If blocked: Update task status, document blocker

---

## Quick Reference

| Phase | Agent | Key Skill | Output |
|-------|-------|-----------|--------|
| Recon | @recon | orchestration | recon_{target}.md |
| Planning | @oracle | planning-with-files | implementation_plan.md |
| Consult | orchestrator | - | User decision |
| Execution | @fixer/@flash | domain-specific | Code changes |
| Verification | @multimodal-looker | - | Verification report |
| Completion | orchestrator | - | Status updates |

---

## Anti-Patterns

❌ **Never skip user consultation** for non-trivial tasks
❌ **Never execute without recon** on unfamiliar code
❌ **Never dispatch without skill context** in prompt
❌ **Never lose track of parallel threads** - always query status
❌ **Never assume completion** - always verify

---

## Integration with MCP

This workflow uses these MCP tools:

- `prep_orchestrator()` - Session start
- `task(action: "list")()` - Find work
- `task(action: "update")()` - Status changes
- `dispatch()` - Record dispatches
- `dispatch(action: "status")()` - Track threads
- `report(action: "recon")()` - Record findings
- `debt(action: "add")()` - Record tech debt
- `task(action: "archive")()` - Completion

See: `global_skills/orchestration/SKILL.md` for MCP details.
