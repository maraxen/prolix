# Research: Agent Memory/State Persistence

## Objective
Design a system to persist agent learnings, preferences, and state across sessions to improve continuity and personalization.

## 1. Types of Persistent State

We identify four distinct categories of state that require persistence:

1.  **User Preferences:** Explicit instructions on how the user likes to work (e.g., "Use Rust", "Be concise").
    *   *Scope:* Global or Project-specific.
    *   *Mutability:* Low frequency updates.
2.  **Project Knowledge/Facts:** specific details about the codebase or architecture that aren't immediately obvious from files (e.g., "We avoid ORM X because of bug Y").
    *   *Scope:* Project-specific.
    *   *Mutability:* Medium.
3.  **Learnings/Insights:** General patterns or solutions discovered by the agent (e.g., "Fix for common error Z").
    *   *Scope:* Global or Project-specific.
4.  **Session/Context State:** The current goal, steps taken, and "scratchpad" notes.
    *   *Scope:* Session (but useful to persist for resuming).

## 2. Storage Mechanisms

### Option A: SQLite (Recommended)
Extend the existing `agent.db` with a `memories` table.
*   **Pros:** ACID compliance, structured data, fast queries, easy integration with existing `agent-infra-mcp`.
*   **Cons:** Requires schema migration; less human-readable than Markdown.

### Option B: Markdown Files (`.agent/memory.md`)
Append facts to a Markdown file.
*   **Pros:** Human-readable, easy to edit, version controllable via Git.
*   **Cons:** Harder for the agent to query precisely; context window pollution if it grows too large.

### Option C: Vector Database (RAG)
Store embeddings of facts.
*   **Pros:** Semantic search allows retrieving relevant memories even with inexact keyword matches.
*   **Cons:** Complexity overhead (embedding model, vector store dependency).

**Recommendation:** **Option A (SQLite)** for structured storage, with a potential **FTS5** (Full-Text Search) virtual table for keyword retrieval. This balances complexity and capability.

## 3. Retrieval Patterns

1.  **Context Injection:**
    *   Fetch high-priority "Preferences" and inject them into the System Prompt at the start of every session.
2.  **Tool-Assisted Retrieval:**
    *   Agent calls `memory_search(query="auth")` when working on a specific task to find relevant past learnings.
3.  **Relevant Context (RAG-lite):**
    *   If using embeddings or FTS, automatically fetch memories related to the current `task_description` and inject them into the context.

## 4. Update Triggers

1.  **Explicit Command:** User says "Remember that ..." -> calls `save_memory`.
2.  **Task Completion:** When completing a task, the agent can optionally save a "Learning" if the solution was novel or difficult.
3.  **Correction:** If the user corrects the agent ("No, don't use X"), the agent should self-trigger a memory update ("Avoid X in this project").

## 5. Privacy and Scope

*   **Project Scope:** `agent.db` is located in `.agent/` within the project root. This ensures project isolation. Sensitive data remains local to the project.
*   **Global Scope:** A separate database (e.g., `~/.gemini/global.db`) would be needed for cross-project learnings.
    *   *Policy:* Default to **Project Scope** to prevent data leakage (e.g., proprietary code patterns) between clients/projects.

## Proposed Schema (SQLite)

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,           -- The actual fact/preference
    category TEXT DEFAULT 'general', -- 'preference', 'fact', 'learning'
    tags TEXT,                       -- JSON array of tags ["rust", "auth"]
    source TEXT,                     -- 'user', 'agent'
    confidence REAL DEFAULT 1.0,     -- 0.0 to 1.0
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Virtual table for efficient text search
CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id');

-- Triggers to keep FTS index in sync
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
  INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
```

## Implementation Roadmap
1.  Add `memories` table to `schema.sql`.
2.  Update `agent-infra-mcp` to support `memory_add`, `memory_search`, `memory_list`.
3.  Update `init` logic to run migrations.
