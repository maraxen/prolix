# Gemini CLI Dispatch Analysis

This document analyzes current Gemini CLI dispatch patterns and identifies automation opportunities for an MCP server.

## 1. Current Patterns Found

| Pattern | Mechanism | Example Command | Source |
|---------|-----------|-----------------|--------|
| **Headless Batch** | Non-interactive execution | `gemini --model "$MODEL" "task"` | `gemini-cli-headless/SKILL.md` |
| **Piped Context** | Stdin for dynamic data | `cat file.txt \| gemini "Explain"` | `gemini-cli-headless/SKILL.md` |
| **Structured Output** | JSON format for parsing | `gemini --output-format json` | `gemini-cli-headless/SKILL.md` |
| **Agent-Mode Dispatch**| Prepend agent system prompt | `gemini ... "Agent: fixer..."` | `orchestration/SKILL.md` |
| **Skill-Aware Dispatch**| Attach skill instructions | `--context global_skills/tdd/SKILL.md` | `orchestration/SKILL.md` |

## 2. Model Selection Logic

Model selection is categorized by task type and agent mode to balance speed and reasoning depth.

| Task Category | Recommended Model | Representative Agents |
|---------------|-------------------|-----------------------|
| **Fast / Recon** | `gemini-3-flash-preview` | `explorer`, `recon`, `flash`, `fixer` |
| **Deep / Planning**| `gemini-3-pro-preview` | `oracle`, `investigator`, `designer`, `librarian` |
| **Unknown** | `auto` | `general`, `evolving-orchestrator` |

**Environment Variables**:
- `GEMINI_MODEL_FAST`: Targets flash models for speed.
- `GEMINI_MODEL_DEEP`: Targets pro models for quality.

## 3. Prompt Construction Template

The following structure is recommended for optimal dispatch prompts:

```markdown
# Agent: {MODE}
[Content of .agent/agents/{MODE}.md]

# Skill: {SKILL_NAME}
[Content of global_skills/{SKILL}/SKILL.md]

# Task Context
Project: {PROJECT_NAME}
Matrix ID: {TASK_ID}
Target Directory: {TARGET_DIR}

# Output Specification
- Write results to: {ARTIFACT_PATH}
- Format: {MARKDOWN|JSON}
- Required Sections: [Summary, Changes, Verification]

# Task
{DETAILED_TASK_DESCRIPTION}
```

## 4. Recommended MCP Tools

To automate these patterns, an MCP server should provide the following tools:

| Tool Name | Parameters (JSON Schema) | Purpose |
|-----------|--------------------------|---------|
| `dispatch_task` | `{"id": "string", "mode": "string", "task": "string", "context": "string[]"}` | High-level tool to load matrix context and dispatch to CLI |
| `select_model` | `{"mode": "string", "complexity": "low\|med\|high"}` | Returns the optimal model ID based on established mapping |
| `format_prompt` | `{"agent": "string", "skills": "string[]", "task": "string"}` | Assembles the standardized prompt components |
| `execute_headless`| `{"prompt": "string", "model": "string", "context_files": "string[]"}` | Low-level wrapper for the `gemini` CLI command |

## 5. Automation Opportunities

1.  **Dynamic Dispatcher (`dispatch.sh`)**: A shell script (or Python tool) that takes a Matrix ID, automatically reads the agent mode and skills from `DEVELOPMENT_MATRIX.md`, constructs the prompt, selects the model, and executes the CLI.
2.  **Context Auto-Packer**: A tool that automatically identifies and attaches relevant context files (e.g., neighboring files in the same directory, or referenced symbols) to reduce manual `--context` flags.
3.  **Result Auto-Matrix**: A post-execution hook that parses the JSON output from a headless run and automatically updates the `DEVELOPMENT_MATRIX.md` status and link to artifacts.
4.  **Session Handoff**: Automating the `--session` flag usage to allow an MCP tool to "pick up where it left off" in long-running tasks.