# MCP Infrastructure Research

> Research artifacts for designing the `agent-infra-mcp` server.

## Purpose

This directory contains reconnaissance outputs that will inform the design and implementation of an MCP (Model Context Protocol) server to automate agent infrastructure operations.

## Analysis Artifacts

| File                          | Focus                        | Status  | Lines |
|-------------------------------|------------------------------|---------|-------|
| `cli_dispatch_analysis.md`    | Gemini CLI dispatch patterns | ✅ DONE | 70    |
| `task_system_analysis.md`     | Task lifecycle automation    | ✅ DONE | 147   |
| `skill_loading_analysis.md`   | Skill & workflow loading     | ✅ DONE | 115   |
| `matrix_ops_analysis.md`      | Dev matrix CRUD operations   | ✅ DONE | 127   |
| `archive_handoff_analysis.md` | Archive & handoff patterns   | ✅ DONE | 57    |

## Key Findings

### CLI Dispatch (gemini-cli-headless skill needs update!)

- No `--context` flag exists - context is via current workspace
- Use `--output-format json` for structured output (MCP can capture directly)
- Use `--approval-mode` for permission control (default/auto_edit/yolo)
- Model selection based on agent mode (flash for recon, pro for planning)

### Proposed MCP Tools

| Tool | Purpose |
|------|---------|
| `matrix_add` | Add task to matrix with auto-ID |
| `matrix_update` | Update status, agents, timestamps |
| `matrix_query` | Query tasks by status/priority/mode |
| `task(action: "create")` | Create task dir + matrix entry |
| `task(action: "archive")` | Move completed task to archive |
| `skill(action: "load")` | Load skill content (full/summary) |
| `dispatch_format` | Construct CLI dispatch prompt |

## Next Steps

1. ~~Synthesize findings into MCP tool specifications~~ ✅ Done (see individual files)
2. **Update gemini-cli-headless skill** - Remove --context refs, add --approval-mode docs
3. Design the MCP server interface
4. Implement `agent-infra-mcp` server

## Related

- Task: `.agent/tasks/mcp001_infrastructure_recon/`
- KI: `standardized_agent_infrastructure` → `orchestration/future_automation_mcp.md`
