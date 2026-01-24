# Skill and Workflow Loading Analysis

## 1. Skill Inventory
The following skills are located in `global_skills/`. Line counts provide an estimate of complexity and detail.

| Skill Name | Purpose | Lines | Key Sections |
|------------|---------|-------|--------------|
| `kaizen` | Continuous improvement & error proofing | 730 | Overview, Four Pillars, Red Flags |
| `planning-with-files` | Persistent memory via markdown files | 211 | Core Pattern, File Purposes, 3-Strike Protocol |
| `ui-ux-pro-max` | UI/UX design intelligence | 228 | Styles, Stacks, UX Guidelines |
| `using-git-worktrees` | Isolated feature work | 217 | Setup, Workflow, Safety Checks |
| `gemini-cli-headless` | CLI automation and model selection | 179 | Model Options, Env Vars, CI Integration |
| `test-driven-development` | TDD methodology | 371 | Red-Green-Refactor, Anti-patterns |
| `systematic-debugging` | Root cause analysis focus | 296 | Iron Law, Four Phases, Red Flags |
| `doc-coauthoring` | Structured doc creation workflow | 375 | Prep, Draft, Review, Polish |
| `orchestration` | Multi-agent coordination patterns | 275 | Dispatch Patterns, Handoff Protocols |
| `senior-architect` | High-level system design | 209 | Patterns, Trade-offs, Decision Logic |
| `senior-fullstack` | Fullstack implementation standards | 209 | Stack Guide, Quality Standards |
| `dev-matrix` | Programmatic matrix management | 137 | ID Generation, Status Updates |
| `agent-modes` | Mode switching and selection | 78 | Decision Tree, Strategy |
| `test-fixing` | Systematic test repair | 119 | Error Grouping, Fix Verification |
| `writing-plans` | Implementation plan creation | 116 | Plan Structure, Review Checklist |
| `requesting-code-review` | Standardized review requests | 105 | PR Templates, Checklists |
| `receiving-code-review` | Processing feedback | 213 | Rigor vs Agreement, Verification |
| `jules-remote` | Remote agent delegation | 165 | Task Isolation, Integration Log |
| `verification-before-completion` | Hard evidence requirement | 139 | Command Verification, Claim Standards |

## 2. SKILL.md Standard Structure
Skills follow a standardized Markdown format with metadata frontmatter:

1.  **Frontmatter (YAML)**:
    - `name`: Identifier for matrix referencing.
    - `description`: One-sentence purpose (used for summaries).
    - `version` (optional): SemVer tracking.
    - `allowed-tools` (optional): Tool whitelist for restricted environments.
2.  **Header**: `# Skill Title`
3.  **Overview**: Core philosophy and principles.
4.  **When to Use**: Activation triggers and context.
5.  **Operational Sections**:
    - `Principles` or `The Four Pillars`
    - `Workflow` or `The Four Phases`
    - `In Practice` (Code examples often with `<Good>` and `<Bad>` tags)
6.  **Safety/Constraint Sections**:
    - `Red Flags`: Indicators of poor application.
    - `The Iron Law`: Non-negotiable rules.
7.  **Integration**: How it interacts with other skills or CLI commands.

## 3. Workflow Inventory
Workflows are located in `.agent/workflows/` and define multi-stage pipelines.

| Workflow Name | Purpose | Primary Agents |
|---------------|---------|----------------|
| `frontend-polish` | UI visual refinement | `flash`, `multimodal-looker` |
| `high-level-review` | Codebase-wide architecture review | `orchestrator`, `oracle` |

**Workflow Structure**:
- Defined by a sequence of stages (e.g., CAPTURE → ANALYZE → FIX → VALIDATE).
- Each stage specifies the **Agent**, **Tool**, **Input**, and **Output**.

## 4. Loading Strategies

### Reference Pattern
- **DEVELOPMENT_MATRIX.md**: Tasks reference skills in the `Skills` column (comma-separated).
- **AGENTS.md**: Defines the protocol for loading these references.

### Full vs. Summary Injection
The `evolving-orchestrator` employs the following strategies when dispatching a task:

1.  **Full Injection**: For the primary skill required (e.g., if `Mode` is `fixer`, the `fixer` prompt is fully loaded).
2.  **Section Extraction**: Key points from `SKILL.md` (Overview, Iron Law, Red Flags) are extracted and injected into the "Skills to Apply" section of the dispatch prompt.
3.  **Template-Based Dispatch**:
    ```markdown
    ## Context
    - Matrix ID: {id}
    - Skills: {extracted_guidance}
    - Research: {summaries}
    - Workflow: {steps}

    ## Task
    {description}
    ```

## 5. MCP skill(action: "load") Tool Spec

To automate skill injection, an MCP tool `skill(action: "load")` should follow this schema:

```json
{
  "name": "skill(action: "load")",
  "description": "Loads and processes skills for agent dispatch context injection.",
  "parameters": {
    "type": "object",
    "properties": {
      "skills": {
        "type": "array",
        "items": { "type": "string" },
        "description": "List of skill names to load (e.g., ['tdd', 'kaizen'])"
      },
      "format": {
        "type": "string",
        "enum": ["full", "summary", "instructions_only"],
        "default": "summary",
        "description": "How to format the skill content for the prompt."
      }
    },
    "required": ["skills"]
  }
}
```

**Loading Logic**:
1. Search `global_skills/{skill}/SKILL.md`.
2. Parse Frontmatter.
3. If `summary`, extract `# Overview` and `## Red Flags`.
4. If `instructions_only`, extract `## Workflow` or `## The Four Phases`.
5. Return a formatted string ready for injection.