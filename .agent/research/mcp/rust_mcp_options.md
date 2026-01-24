# Rust MCP Server Options Research

> **Date**: 2026-01-20
> **Purpose**: Evaluate Rust implementations for the agent-infra-mcp server

## Summary

There are **three main Rust MCP implementations** available:

### 1. Official Rust SDK (`modelcontextprotocol/rust-sdk`) ⭐ Recommended

- **GitHub**: <https://github.com/modelcontextprotocol/rust-sdk>
- **Crates**:
  - `rmcp` - Core protocol implementation
  - `rmcp-macros` - Proc macros for generating tool implementations
- **Status**: Official, maintained by Anthropic
- **Pros**: Official support, well-documented, likely to track spec changes
- **Cons**: May be newer/less battle-tested

### 2. MCPR (Complete Implementation)

- **Source**: Available on GitHub and crates.io
- **Features**:
  - Server and client stubs generation
  - Transport layer implementations
  - Command-line utilities
- **Status**: Community maintained
- **Pros**: Complete feature set
- **Cons**: Community maintained (sustainability?)

### 3. mcp-sdk (Minimalistic)

- **Crates.io**: `mcp-sdk`
- **Design**: Minimalistic, agile, easy to understand
- **Compatibility**: Tested with Claude Desktop
- **Pros**: Simple, lightweight
- **Cons**: May lack advanced features

## Protocol Details

- **Transport**: JSON-RPC based
- **Purpose**: Universal interface for reading files, executing functions, handling contextual prompts
- **Use Cases**: Connect to GitHub, Google Drive, Slack, file systems, etc.

## Recommendation

For `agent-infra-mcp`:

1. **Start with official `rmcp` SDK** - Best long-term support
2. Use **stdio transport** for local CLI integration
3. Consider **HTTP transport** for future Jules/remote dispatch

## Integration Pattern

```rust
// Example: MCP server structure
use rmcp::{Server, Tool, ToolResult};

#[tool]
async fn matrix_add(description: String, priority: Priority) -> ToolResult {
    // Add to DEVELOPMENT_MATRIX.md
}

#[tool]
async fn task(action: "create")(id: String, name: String) -> ToolResult {
    // Create task directory and README
}
```

## Additional Resources

- Official MCP Docs: <https://anthropic.com/mcp>
- Rust SDK Docs: <https://docs.rs/rmcp>
- Wikipedia: Model Context Protocol (JSON-RPC based)

## Next Steps

1. Review `rmcp` crate documentation
2. Set up basic server skeleton
3. Implement matrix operations first (highest value)
