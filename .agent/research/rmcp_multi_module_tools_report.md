# Research Report: RMCP Multi-Module Tool Patterns

**Date**: 2026-01-20  
**Author**: Orchestrator (with Gemini 3-pro dispatch + web research)  
**Status**: Complete

## Dispatch Metadata

| Field | Value |
|-------|-------|
| Task ID | rmcp_multi_module_research |
| Dispatched | 2026-01-20T19:07:32-05:00 |
| Model | gemini-3-pro-preview |
| Agent Mode | researcher |
| Completion | 2026-01-20T19:10:00-05:00 |
| Outcome | SUCCESS (web research + CLI partial) |

> **Note**: Gemini CLI dispatch was terminated early after web research yielded the key finding (tool_router combination pattern). CLI agent was limited to local file exploration since it lacks shell command and web search tools in current config.

---

## Summary

**Yes, rmcp supports splitting tools across multiple modules.** The key mechanism is the `#[tool_router]` macro with named routers that can be **combined using the `+` operator**.

---

## Key Finding: Router Combination Pattern

From the [rmcp docs](https://docs.rs/rmcp/latest/rmcp/attr.tool_router.html):

```rust
mod a {
    #[tool_router(router = tool_router_a, vis = "pub")]
    impl MyToolHandler {
        #[tool]
        fn my_tool_a() { }
    }
}

mod b {
    #[tool_router(router = tool_router_b, vis = "pub")]
    impl MyToolHandler {
        #[tool]
        fn my_tool_b() { }
    }
}

impl MyToolHandler {
    fn new() -> Self {
        Self {
            // Combine routers with + operator!
            tool_router: a::tool_router_a() + b::tool_router_b(),
        }
    }
}
```

### Key Points

1. **Named Routers**: Use `router = tool_router_NAME` to give each impl block's router a unique name
2. **Visibility**: Set `vis = "pub"` to export the router function from submodules
3. **Combination**: Use `+` operator to merge multiple `ToolRouter` instances
4. **Handler Field**: Store the combined router in a field (typically `tool_router: ToolRouter<Self>`)

---

## Current Architecture Analysis

Our `server.rs` uses `#[tool(tool_box)]` on a single impl block. We have two architectural options:

### Option A: Migrate to `#[tool_router]` Pattern (Recommended)

**Pros:**

- Native multi-module support
- Clean separation of concerns
- Documented, supported pattern

**Cons:**

- Requires refactoring the impl block
- Need to add `ToolRouter<Self>` field to struct

### Option B: Keep Single `#[tool(tool_box)]` + Helper Modules

**Pros:**

- Minimal changes to server.rs
- Extract request/response types and helper functions only

**Cons:**

- Tool implementations stay in one file
- Doesn't fully solve the file length issue

---

## Recommended Implementation Plan

### Phase 1: Preparation

1. Add `ToolRouter<Self>` field to `AgentInfraServer` struct
2. Verify current code compiles with `#[tool_router]` instead of `#[tool(tool_box)]`

### Phase 2: Extract First Module (Spike)

1. Create `src/tools/debt.rs` with `debt(action: "add")` and `debt(action: "list")`
2. Use `#[tool_router(router = debt_router, vis = "pub")]`
3. Verify combination works in `AgentInfraServer::new()`

### Phase 3: Full Modularization

```
src/
├── server.rs              # Core struct, ServerHandler impl, main router combo
├── tools/
│   ├── mod.rs             # Re-exports tool routers
│   ├── workspace.rs       # workspace_handshake, agent_init, workspace_info
│   ├── task.rs            # task(action: "list"), task(action: "update"), task(action: "create"), task(action: "archive")
│   ├── workflow.rs        # workflow(action: "list"), workflow(action: "trigger")
│   ├── dispatch.rs        # dispatch, dispatch(action: "status")
│   ├── prompt.rs          # prompt(action: "list"), prompt(action: "invoke")
│   ├── skill.rs           # skill(action: "list"), skill(action: "load")
│   ├── debt.rs            # debt(action: "add"), debt(action: "list")
│   ├── research.rs        # report(action: "recon"), report(action: "research")
│   ├── orchestrator.rs    # prep_orchestrator
│   └── admin.rs           # config(action: "backup"), config(action: "export")
```

### Final RError during web search for query "rust modelcontextprotocol sdk split tools multiple files": The user aborted a request. GaxiosError: The user aborted a request

    at Gaxios._request (/opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/gaxios/build/src/gaxios.js:149:19)
    at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
    at async OAuth2Client.requestAsync (/opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/build/src/auth/oauth2client.js:429:18)
    at async CodeAssistServer.requestPost (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:133:21)
    at async CodeAssistServer.generateContent (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:46:26)
    at async file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/loggingContentGenerator.js:94:34
    at async retryWithBackoff (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/utils/retry.js:108:28)
    at async GeminiClient.generateContent (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/client.js:625:28)
    at async WebSearchToolInvocation.execute (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/tools/web-search.js:25:30)
    at async executeToolWithHooks (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/coreToolHookTriggers.js:269:22) {
  config: {
    url: '<https://cloudcode-pa.googleapis.com/v1internal:generateContent>',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'GeminiCLI/0.25.0-preview.0/gemini-3-pro-preview (darwin; arm64) google-api-nodejs-client/9.15.1',
      Authorization: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      'x-goog-api-client': 'gl-node/24.1.0',
      Accept: 'application/json'
    },
    responseType: 'json',
    body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
    signal: AbortSignal { aborted: true },
    paramsSerializer: [Function: paramsSerializer],
    validateStatus: [Function: validateStatus],
    errorRedactor: [Function: defaultErrorRedactor]
  },
  response: undefined,
  error: AbortError: The user aborted a request.
      at abort (/opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/node-fetch/lib/index.js:1458:16)
      at AbortSignal.abortAndFinalize (/opt/homebrew/lib/node_modules/@google/gemini-cli/node_modules/node-fetch/lib/index.js:1473:4)
      at [nodejs.internal.kHybridDispatch] (node:internal/event_target:827:20)
      at AbortSignal.dispatchEvent (node:internal/event_target:762:26)
      at runAbort (node:internal/abort_controller:488:10)
      at abortSignal (node:internal/abort_controller:459:3)
      at AbortController.abort (node:internal/abort_controller:507:5)
      at ReadStream.keypressHandler (file:///opt/homebrew/lib/node_modules/@google/gemini-cli/dist/src/nonInteractiveCli.js:77:37)
      at ReadStream.emit (node:events:507:28)
      at emitKeys (node:internal/readline/utils:371:14) {
    type: 'aborted'
  },
  Symbol(gaxios-gaxios-error): '6.7.1'
}
Operation cancelled.[ERROR] Operation cancelled.
