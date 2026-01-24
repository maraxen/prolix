# Gemini CLI Headless Mode Research

## Overview

The Gemini CLI can be used in "headless" or non-interactive mode for automation, scripting, and integration with other tools (like the Dev Matrix).

## Key Capabilities

1. **Standard Input Piping**:
   The CLI accepts input from stdin, allowing scripts to pipe prompts directly to it.

   ```bash
   echo "Analyze this file: $(cat file.txt)" | gemini
   ```

2. **Output Formatting**:
   The `--output-format` flag is critical for automation.
   - `text`: Standard human-readable output (default).
   - `json`: Structured JSON output, easier for scripts to parse.
   - `stream-json`: Streaming JSON for real-time processing.

   ```bash
   gemini --output-format json < prompt.txt > response.json
   ```

3. **Session Management**:
   The CLI supports session persistence, allowing multi-turn conversations in scripts.
   - `--session-id`: Resume a specific session.
   - `--list-sessions`: See active sessions.

## Automation Patterns

### One-shot Task

```bash
gemini "Refactor this code" --context file.js --output-format text > output.txt
```

### Batch Processing

Write a loop in bash/python to iterate over files and pipe them to gemini.

```bash
for file in src/*.js; do
  echo "Review $file" | gemini --context "$file" --output-format json >> review_log.json
done
```

## Limitations

- **Statefulness**: Unless a session ID is managed, each invocation is stateless.
- **Context Window**: Be mindful of passing too many files via `--context`.
- **Auth**: Requires valid credentials (usually already set up in `~/.gemini/`).

## Recommendation

Create a `gemini-cli-headless` skill to standardize these patterns for agents.
