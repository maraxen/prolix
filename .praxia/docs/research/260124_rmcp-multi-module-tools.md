# Research: RMCP Multi-Module Tool Patterns

## Objective

Investigate how the `rmcp` Rust SDK supports splitting MCP tool implementations across multiple source files/modules while maintaining the `#[tool(tool_box)]` macro functionality.

## Context

Our `server.rs` has grown to ~1,162 lines with 22+ tool implementations. We want to modularize into domain-specific files (e.g., `tools/task.rs`, `tools/skill.rs`) while preserving the rmcp tool registration behavior.

## Questions to Answer

1. **Can multiple `#[tool(tool_box)]` impl blocks exist on the same struct?**
   - If so, can they be in different modules?

2. **Are there trait-based patterns for tool delegation?**
   - Can we define tools in separate traits, then impl those traits on AgentInfraServer?

3. **How does rmcp's `ToolBox` derive work?**
   - What are the requirements for tool discovery/registration?
   - Does it scan all impl blocks or only the one with the macro?

4. **What are common patterns in rmcp examples/tests for multi-file servers?**
   - Check the rmcp repo's examples/ and tests/ directories

5. **Are there limitations or gotchas with splitting tools?**
   - Ordering dependencies?
   - Visibility requirements?
   - Macro expansion issues?

## Sources to Check

- [ ] rmcp crate documentation (docs.rs/rmcp)
- [ ] rmcp GitHub repository (modelcontextprotocol/rust-sdk or equivalent)
- [ ] Any existing MCP servers using rmcp with modular structure
- [ ] The `#[tool]` and `#[tool(tool_box)]` macro source code if accessible

## Deliverables

1. Summary of supported patterns
2. Recommended approach for our use case
3. Code example showing the pattern
4. Any gotchas or limitations discovered
