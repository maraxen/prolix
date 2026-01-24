# Research: Tags Feature for Tasks and Tech Debt

**Date**: 2026-01-21  
**Researcher**: Antigravity Agent  
**Task ID**: 260120204825  
**Dispatch ID**: d260120210509

---

## Executive Summary

This research proposes a flexible tagging system for `tasks` and `tech_debt` tables using **JSON arrays** stored in TEXT columns. This approach balances simplicity, performance, and SQLite compatibility while enabling powerful filtering, autocomplete, and tag management features.

**Key Design Decisions**:

1. **JSON Array Storage**: Store tags as JSON arrays in TEXT columns (e.g., `["frontend", "urgent", "refactor"]`)
2. **SQLite JSON Functions**: Leverage `json_each()` and `json_extract()` for querying
3. **Composite Indexes**: Create indexes on frequently queried tag combinations
4. **MCP Tool Extensions**: Add `tags` parameter to `task(action: "list")`/`debt(action: "list")`, new `tag_list` tool
5. **Migration Strategy**: Add nullable `tags` column, populate with defaults based on existing fields

**Benefits**:

- Simple schema (single column per table)
- Flexible (unlimited tags per item)
- Performant (with proper indexing)
- SQLite-native (no external dependencies)
- Backward compatible (nullable column)

---

## 1. Schema Design Options

### 1.1 Option A: JSON Array in TEXT Column (RECOMMENDED)

**Schema**:

```sql
ALTER TABLE tasks ADD COLUMN tags TEXT;
ALTER TABLE tech_debt ADD COLUMN tags TEXT;

-- Add JSON validation constraints
ALTER TABLE tasks ADD CONSTRAINT valid_tags CHECK (tags IS NULL OR json_valid(tags));
ALTER TABLE tech_debt ADD CONSTRAINT valid_tags CHECK (tags IS NULL OR json_valid(tags));
```

**Example Data**:

```sql
-- Task with tags
INSERT INTO tasks (id, description, tags) VALUES 
  ('abc123', 'Refactor auth module', '["backend", "refactor", "security"]');

-- Tech debt with tags
INSERT INTO tech_debt (title, category, tags) VALUES 
  ('Fix schema validation', 'schema', '["bug", "high-priority", "database"]');
```

**Pros**:

- ✅ Simple schema (single column)
- ✅ Flexible (unlimited tags, no schema changes to add tags)
- ✅ SQLite-native (JSON functions built-in since 3.38.0)
- ✅ Easy to migrate (add column, populate with defaults)
- ✅ Compact storage (JSON array is space-efficient)

**Cons**:

- ⚠️ Requires JSON functions for querying (slightly more complex SQL)
- ⚠️ No foreign key constraints (typos possible, e.g., "frontent" vs "frontend")
- ⚠️ Index strategy is critical for performance

**Query Examples**:

```sql
-- Find tasks with "frontend" tag
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) 
  WHERE json_each.value = 'frontend'
);

-- Find tasks with "frontend" OR "backend" tag
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) 
  WHERE json_each.value IN ('frontend', 'backend')
);

-- Find tasks with "frontend" AND "urgent" tags
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) WHERE json_each.value = 'frontend'
) AND EXISTS (
  SELECT 1 FROM json_each(tasks.tags) WHERE json_each.value = 'urgent'
);
```

### 1.2 Option B: Junction Table (Normalized)

**Schema**:

```sql
CREATE TABLE task_tags (
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (task_id, tag)
);

CREATE TABLE debt_tags (
    debt_id INTEGER NOT NULL REFERENCES tech_debt(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (debt_id, tag)
);

CREATE INDEX idx_task_tags_tag ON task_tags(tag);
CREATE INDEX idx_debt_tags_tag ON debt_tags(tag);
```

**Example Data**:

```sql
-- Task with tags
INSERT INTO tasks (id, description) VALUES ('abc123', 'Refactor auth module');
INSERT INTO task_tags (task_id, tag) VALUES 
  ('abc123', 'backend'),
  ('abc123', 'refactor'),
  ('abc123', 'security');
```

**Pros**:

- ✅ Normalized (follows relational best practices)
- ✅ Easy to query (standard SQL joins)
- ✅ Can add foreign key to `tags` table (enforce valid tags)
- ✅ Efficient for large tag sets (no JSON parsing)

**Cons**:

- ❌ More complex schema (2 new tables)
- ❌ More complex queries (requires JOINs)
- ❌ More storage overhead (row per tag)
- ❌ Harder to migrate (need to populate junction tables)

**Query Examples**:

```sql
-- Find tasks with "frontend" tag
SELECT t.* FROM tasks t
JOIN task_tags tt ON t.id = tt.task_id
WHERE tt.tag = 'frontend';

-- Find tasks with "frontend" OR "backend" tag
SELECT DISTINCT t.* FROM tasks t
JOIN task_tags tt ON t.id = tt.task_id
WHERE tt.tag IN ('frontend', 'backend');

-- Find tasks with "frontend" AND "urgent" tags
SELECT t.* FROM tasks t
WHERE EXISTS (SELECT 1 FROM task_tags WHERE task_id = t.id AND tag = 'frontend')
  AND EXISTS (SELECT 1 FROM task_tags WHERE task_id = t.id AND tag = 'urgent');
```

### 1.3 Option C: Full-Text Search (FTS5)

**Schema**:

```sql
-- Create FTS5 virtual table for tasks
CREATE VIRTUAL TABLE tasks_fts USING fts5(
    id UNINDEXED,
    description,
    tags,
    content=tasks,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER tasks_fts_insert AFTER INSERT ON tasks BEGIN
    INSERT INTO tasks_fts(rowid, id, description, tags) 
    VALUES (new.rowid, new.id, new.description, new.tags);
END;

CREATE TRIGGER tasks_fts_update AFTER UPDATE ON tasks BEGIN
    UPDATE tasks_fts SET description = new.description, tags = new.tags 
    WHERE rowid = new.rowid;
END;

CREATE TRIGGER tasks_fts_delete AFTER DELETE ON tasks BEGIN
    DELETE FROM tasks_fts WHERE rowid = old.rowid;
END;
```

**Example Data**:

```sql
-- Task with tags (stored as space-separated string)
INSERT INTO tasks (id, description, tags) VALUES 
  ('abc123', 'Refactor auth module', 'backend refactor security');
```

**Pros**:

- ✅ Extremely fast full-text search
- ✅ Supports fuzzy matching, stemming, ranking
- ✅ Good for large datasets (millions of rows)

**Cons**:

- ❌ Overkill for simple tag filtering
- ❌ Requires triggers to keep FTS in sync
- ❌ More storage overhead (FTS index)
- ❌ Tags must be space-separated strings (not structured)
- ❌ Complex to maintain

**Query Examples**:

```sql
-- Find tasks with "frontend" tag (full-text search)
SELECT t.* FROM tasks t
JOIN tasks_fts fts ON t.rowid = fts.rowid
WHERE tasks_fts MATCH 'tags:frontend';
```

### 1.4 Recommendation: Option A (JSON Array)

**Rationale**:

- **Simplicity**: Single column, minimal schema changes
- **Flexibility**: Unlimited tags, easy to add/remove
- **Performance**: Adequate for expected dataset size (hundreds to thousands of tasks)
- **SQLite Compatibility**: JSON functions are built-in and well-supported
- **Migration**: Easy to add column and populate with defaults

**When to Reconsider**:

- If dataset grows to 100K+ tasks/debt items, consider Option B (junction table)
- If full-text search is needed (e.g., search task descriptions), consider Option C (FTS5)

---

## 2. Query Patterns

### 2.1 Single Tag Filter

**Use Case**: Find all tasks tagged "frontend"

**SQL**:

```sql
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) 
  WHERE json_each.value = 'frontend'
);
```

**Rust (db.rs)**:

```rust
pub fn query_tasks_by_tag(&self, tag: &str) -> Result<Vec<DbTask>> {
    let mut stmt = self.conn.prepare(
        "SELECT id, status, priority, difficulty, mode, description, tags, created_at, updated_at, archived_at 
         FROM tasks 
         WHERE EXISTS (
           SELECT 1 FROM json_each(tasks.tags) 
           WHERE json_each.value = ?1
         )"
    )?;
    
    let iter = stmt.query_map(params![tag], |row| {
        Ok(DbTask {
            id: row.get(0)?,
            status: row.get(1)?,
            priority: row.get(2)?,
            difficulty: row.get(3)?,
            mode: row.get(4)?,
            description: row.get(5)?,
            tags: parse_json_array(row.get::<_, Option<String>>(6)?),
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
            archived_at: row.get(9)?,
            // ... other fields
        })
    })?;
    
    // Collect results...
}
```

### 2.2 OR Filtering (Any Tag Matches)

**Use Case**: Find all tasks tagged "frontend" OR "backend"

**SQL**:

```sql
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) 
  WHERE json_each.value IN ('frontend', 'backend')
);
```

**Rust (db.rs)**:

```rust
pub fn query_tasks_by_tags_or(&self, tags: &[&str]) -> Result<Vec<DbTask>> {
    let placeholders = tags.iter().enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");
    
    let query = format!(
        "SELECT * FROM tasks 
         WHERE EXISTS (
           SELECT 1 FROM json_each(tasks.tags) 
           WHERE json_each.value IN ({})
         )",
        placeholders
    );
    
    let mut stmt = self.conn.prepare(&query)?;
    let params: Vec<&dyn rusqlite::ToSql> = tags.iter()
        .map(|t| t as &dyn rusqlite::ToSql)
        .collect();
    
    // Execute query...
}
```

### 2.3 AND Filtering (All Tags Must Match)

**Use Case**: Find all tasks tagged "frontend" AND "urgent"

**SQL**:

```sql
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) WHERE json_each.value = 'frontend'
) AND EXISTS (
  SELECT 1 FROM json_each(tasks.tags) WHERE json_each.value = 'urgent'
);
```

**Rust (db.rs)**:

```rust
pub fn query_tasks_by_tags_and(&self, tags: &[&str]) -> Result<Vec<DbTask>> {
    let conditions = tags.iter().enumerate()
        .map(|(i, _)| format!(
            "EXISTS (SELECT 1 FROM json_each(tasks.tags) WHERE json_each.value = ?{})",
            i + 1
        ))
        .collect::<Vec<_>>()
        .join(" AND ");
    
    let query = format!("SELECT * FROM tasks WHERE {}", conditions);
    
    let mut stmt = self.conn.prepare(&query)?;
    let params: Vec<&dyn rusqlite::ToSql> = tags.iter()
        .map(|t| t as &dyn rusqlite::ToSql)
        .collect();
    
    // Execute query...
}
```

### 2.4 List All Unique Tags

**Use Case**: Get all unique tags across all tasks (for autocomplete)

**SQL**:

```sql
SELECT DISTINCT json_each.value AS tag
FROM tasks, json_each(tasks.tags)
WHERE tasks.tags IS NOT NULL
ORDER BY tag;
```

**Rust (db.rs)**:

```rust
pub fn list_all_task_tags(&self) -> Result<Vec<String>> {
    let mut stmt = self.conn.prepare(
        "SELECT DISTINCT json_each.value AS tag
         FROM tasks, json_each(tasks.tags)
         WHERE tasks.tags IS NOT NULL
         ORDER BY tag"
    )?;
    
    let iter = stmt.query_map([], |row| row.get(0))?;
    let mut tags = Vec::new();
    for tag in iter {
        tags.push(tag?);
    }
    Ok(tags)
}
```

### 2.5 Tag Frequency (Tag Cloud)

**Use Case**: Get tag usage counts (for tag cloud or analytics)

**SQL**:

```sql
SELECT json_each.value AS tag, COUNT(*) AS count
FROM tasks, json_each(tasks.tags)
WHERE tasks.tags IS NOT NULL
GROUP BY tag
ORDER BY count DESC;
```

**Rust (db.rs)**:

```rust
#[derive(Debug, serde::Serialize)]
pub struct TagCount {
    pub tag: String,
    pub count: usize,
}

pub fn get_task_tag_counts(&self) -> Result<Vec<TagCount>> {
    let mut stmt = self.conn.prepare(
        "SELECT json_each.value AS tag, COUNT(*) AS count
         FROM tasks, json_each(tasks.tags)
         WHERE tasks.tags IS NOT NULL
         GROUP BY tag
         ORDER BY count DESC"
    )?;
    
    let iter = stmt.query_map([], |row| {
        Ok(TagCount {
            tag: row.get(0)?,
            count: row.get(1)?,
        })
    })?;
    
    let mut counts = Vec::new();
    for count in iter {
        counts.push(count?);
    }
    Ok(counts)
}
```

### 2.6 Autocomplete (Prefix Search)

**Use Case**: Autocomplete tag input (e.g., user types "fro", suggest "frontend")

**SQL**:

```sql
SELECT DISTINCT json_each.value AS tag
FROM tasks, json_each(tasks.tags)
WHERE tasks.tags IS NOT NULL 
  AND json_each.value LIKE 'fro%'
ORDER BY tag
LIMIT 10;
```

**Rust (db.rs)**:

```rust
pub fn autocomplete_task_tags(&self, prefix: &str, limit: usize) -> Result<Vec<String>> {
    let mut stmt = self.conn.prepare(
        "SELECT DISTINCT json_each.value AS tag
         FROM tasks, json_each(tasks.tags)
         WHERE tasks.tags IS NOT NULL 
           AND json_each.value LIKE ?1
         ORDER BY tag
         LIMIT ?2"
    )?;
    
    let pattern = format!("{}%", prefix);
    let iter = stmt.query_map(params![pattern, limit], |row| row.get(0))?;
    
    let mut tags = Vec::new();
    for tag in iter {
        tags.push(tag?);
    }
    Ok(tags)
}
```

---

## 3. MCP Tool Extensions

### 3.1 Extend `task(action: "list")` Tool

**Current Signature**:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TaskListRequest {
    pub status: Option<String>,
    pub priority: Option<String>,
    pub mode: Option<String>,
}
```

**New Signature** (with tags):

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TaskListRequest {
    pub status: Option<String>,
    pub priority: Option<String>,
    pub mode: Option<String>,
    
    // NEW: Tag filtering
    pub tags: Option<Vec<String>>,        // Tags to filter by
    pub tag_mode: Option<String>,         // "any" (OR) or "all" (AND), default: "any"
}
```

**Implementation**:

```rust
#[tool(description = "List tasks from the development matrix")]
async fn task(action: "list")(&self, params: Parameters<TaskListRequest>) -> Result<CallToolResult, McpError> {
    let req = params.into_inner();
    let db_lock = self.db.lock().map_err(|e| McpError::internal_error(format!("Lock failed: {}", e), None))?;
    let db = db_lock.as_ref().ok_or_else(|| McpError::invalid_request("No active workspace", None))?;
    
    let tasks = if let Some(tags) = req.tags {
        let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
        match req.tag_mode.as_deref() {
            Some("all") => db.query_tasks_by_tags_and(&tag_refs, req.status.as_deref(), req.priority.as_deref(), req.mode.as_deref())?,
            _ => db.query_tasks_by_tags_or(&tag_refs, req.status.as_deref(), req.priority.as_deref(), req.mode.as_deref())?,
        }
    } else {
        db.query_tasks(req.status.as_deref(), req.priority.as_deref(), req.mode.as_deref())?
    };
    
    let json = serde_json::to_string_pretty(&tasks)
        .map_err(|e| McpError::internal_error(format!("Serialize failed: {}", e), None))?;
    
    Ok(CallToolResult::success(vec![Content::text(json)]))
}
```

### 3.2 Extend `debt(action: "list")` Tool

**Current Signature**:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct DebtListRequest {
    pub status: Option<String>,
    pub category: Option<String>,
}
```

**New Signature** (with tags):

```rust
#[derive(Debug, serde::Deserialize)]
pub struct DebtListRequest {
    pub status: Option<String>,
    pub category: Option<String>,
    
    // NEW: Tag filtering
    pub tags: Option<Vec<String>>,
    pub tag_mode: Option<String>,  // "any" or "all", default: "any"
}
```

**Implementation**: Similar to `task(action: "list")` extension.

### 3.3 New `tag_list` Tool

**Purpose**: List all unique tags, optionally filtered by entity type (tasks, tech_debt).

**Signature**:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TagListRequest {
    /// Entity type: "tasks", "tech_debt", or "all"
    /// Default: "all"
    pub entity: Option<String>,
    
    /// Include tag usage counts
    /// Default: false
    pub include_counts: Option<bool>,
    
    /// Autocomplete prefix filter
    pub prefix: Option<String>,
    
    /// Limit number of results
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
pub struct TagListResponse {
    pub tags: Vec<TagInfo>,
}

#[derive(Debug, serde::Serialize)]
pub struct TagInfo {
    pub tag: String,
    pub count: Option<usize>,  // Only if include_counts=true
    pub entities: Vec<String>, // ["tasks", "tech_debt"]
}
```

**Implementation**:

```rust
#[tool(description = "List all unique tags across tasks and tech debt")]
async fn tag_list(&self, params: Parameters<TagListRequest>) -> Result<CallToolResult, McpError> {
    let req = params.into_inner();
    let db_lock = self.db.lock().map_err(|e| McpError::internal_error(format!("Lock failed: {}", e), None))?;
    let db = db_lock.as_ref().ok_or_else(|| McpError::invalid_request("No active workspace", None))?;
    
    let entity = req.entity.as_deref().unwrap_or("all");
    let include_counts = req.include_counts.unwrap_or(false);
    let limit = req.limit.unwrap_or(100);
    
    let tags = match entity {
        "tasks" => db.list_task_tags(req.prefix.as_deref(), include_counts, limit)?,
        "tech_debt" => db.list_debt_tags(req.prefix.as_deref(), include_counts, limit)?,
        _ => db.list_all_tags(req.prefix.as_deref(), include_counts, limit)?,
    };
    
    let response = TagListResponse { tags };
    let json = serde_json::to_string_pretty(&response)
        .map_err(|e| McpError::internal_error(format!("Serialize failed: {}", e), None))?;
    
    Ok(CallToolResult::success(vec![Content::text(json)]))
}
```

### 3.4 Extend `task(action: "create")` Tool

**Current Signature**:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TaskCreateRequest {
    pub description: String,
    pub status: Option<String>,
    pub priority: Option<String>,
    pub difficulty: Option<String>,
    pub mode: Option<String>,
    pub skills: Option<Vec<String>>,
    pub research: Option<Vec<String>>,
    pub workflows: Option<Vec<String>>,
    pub agents: Option<Vec<String>>,
}
```

**New Signature** (with tags):

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TaskCreateRequest {
    pub description: String,
    pub status: Option<String>,
    pub priority: Option<String>,
    pub difficulty: Option<String>,
    pub mode: Option<String>,
    pub skills: Option<Vec<String>>,
    pub research: Option<Vec<String>>,
    pub workflows: Option<Vec<String>>,
    pub agents: Option<Vec<String>>,
    
    // NEW: Tags
    pub tags: Option<Vec<String>>,
}
```

**Implementation**: Serialize tags to JSON array and store in `tags` column.

### 3.5 Extend `task(action: "update")` Tool

**Current Signature**:

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TaskUpdateRequest {
    pub id: String,
    pub status: Option<String>,
    pub priority: Option<String>,
    pub description: Option<String>,
    pub difficulty: Option<String>,
    pub mode: Option<String>,
}
```

**New Signature** (with tags):

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TaskUpdateRequest {
    pub id: String,
    pub status: Option<String>,
    pub priority: Option<String>,
    pub description: Option<String>,
    pub difficulty: Option<String>,
    pub mode: Option<String>,
    
    // NEW: Tags (replaces existing tags if provided)
    pub tags: Option<Vec<String>>,
}
```

**Implementation**: Serialize tags to JSON array and update `tags` column.

---

## 4. Migration Strategy

### 4.1 Schema Migration

**File**: `.agent/migrations/003_add_tags.sql`

```sql
-- Add tags column to tasks table
ALTER TABLE tasks ADD COLUMN tags TEXT;

-- Add tags column to tech_debt table
ALTER TABLE tech_debt ADD COLUMN tags TEXT;

-- Add JSON validation constraints
-- Note: SQLite doesn't support adding constraints to existing tables directly
-- We'll validate at the application layer instead

-- Create indexes for tag queries
CREATE INDEX idx_tasks_tags ON tasks(tags) WHERE tags IS NOT NULL;
CREATE INDEX idx_tech_debt_tags ON tech_debt(tags) WHERE tags IS NOT NULL;
```

**Migration Logic** (in `db.rs`):

```rust
pub fn migrate_003_add_tags(&self) -> Result<()> {
    let migration = fs::read_to_string(".agent/migrations/003_add_tags.sql")?;
    self.conn.execute_batch(&migration)?;
    Ok(())
}
```

### 4.2 Default Tag Population

**Strategy**: Populate tags based on existing fields to provide initial categorization.

**Rules**:

1. **Tasks**:
   - Add tag based on `mode` (e.g., mode="researcher" → tag="research")
   - Add tag based on `priority` (e.g., priority="P1" → tag="high-priority")
   - Add tag based on `difficulty` (e.g., difficulty="hard" → tag="complex")
   - Add tag based on `status` (e.g., status="BLOCKED" → tag="blocked")

2. **Tech Debt**:
   - Add tag based on `category` (e.g., category="schema" → tag="database")
   - Add tag based on `status` (e.g., status="open" → tag="unresolved")
   - Add tag based on `impact` (if high impact → tag="critical")

**SQL**:

```sql
-- Populate task tags based on existing fields
UPDATE tasks SET tags = json_array(
    CASE mode
        WHEN 'researcher' THEN 'research'
        WHEN 'fixer' THEN 'bugfix'
        WHEN 'orchestrator' THEN 'orchestration'
        ELSE mode
    END,
    CASE priority
        WHEN 'P1' THEN 'high-priority'
        WHEN 'P2' THEN 'medium-priority'
        WHEN 'P3' THEN 'low-priority'
    END,
    CASE difficulty
        WHEN 'hard' THEN 'complex'
        WHEN 'easy' THEN 'simple'
    END
) WHERE tags IS NULL;

-- Populate tech debt tags based on existing fields
UPDATE tech_debt SET tags = json_array(
    category,
    CASE status
        WHEN 'open' THEN 'unresolved'
        WHEN 'in_progress' THEN 'active'
        WHEN 'resolved' THEN 'fixed'
    END
) WHERE tags IS NULL;
```

**Rust (db.rs)**:

```rust
pub fn populate_default_tags(&self) -> Result<()> {
    // Populate task tags
    self.conn.execute(
        "UPDATE tasks SET tags = json_array(
            CASE mode
                WHEN 'researcher' THEN 'research'
                WHEN 'fixer' THEN 'bugfix'
                WHEN 'orchestrator' THEN 'orchestration'
                ELSE mode
            END,
            CASE priority
                WHEN 'P1' THEN 'high-priority'
                WHEN 'P2' THEN 'medium-priority'
                WHEN 'P3' THEN 'low-priority'
            END
        ) WHERE tags IS NULL",
        [],
    )?;
    
    // Populate tech debt tags
    self.conn.execute(
        "UPDATE tech_debt SET tags = json_array(
            category,
            CASE status
                WHEN 'open' THEN 'unresolved'
                WHEN 'in_progress' THEN 'active'
                WHEN 'resolved' THEN 'fixed'
            END
        ) WHERE tags IS NULL",
        [],
    )?;
    
    Ok(())
}
```

### 4.3 Migration Workflow

**Steps**:

1. Run schema migration (add `tags` column)
2. Populate default tags based on existing fields
3. Update MCP tools to support tag filtering
4. Update Rust structs (`DbTask`, `DbTechDebt`) to include `tags` field
5. Test tag queries and filtering
6. Document tag conventions in `.agent/README.md`

**Rollback Plan**:

- If migration fails, drop `tags` column: `ALTER TABLE tasks DROP COLUMN tags;`
- If default population fails, leave `tags` as NULL (no data loss)

---

## 5. SQLite Considerations

### 5.1 JSON Functions

**Required SQLite Version**: 3.38.0+ (for full JSON support)

**Key Functions**:

- `json_array(...)`: Create JSON array
- `json_each(json_text)`: Expand JSON array to rows
- `json_extract(json_text, path)`: Extract value from JSON
- `json_valid(json_text)`: Validate JSON syntax

**Compatibility Check**:

```rust
pub fn check_json_support(&self) -> Result<bool> {
    let version: String = self.conn.query_row("SELECT sqlite_version()", [], |row| row.get(0))?;
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() < 2 {
        return Ok(false);
    }
    let major: u32 = parts[0].parse().unwrap_or(0);
    let minor: u32 = parts[1].parse().unwrap_or(0);
    Ok(major > 3 || (major == 3 && minor >= 38))
}
```

### 5.2 Index Strategies

**Challenge**: SQLite doesn't support indexing JSON array elements directly.

**Solution 1: Partial Index** (recommended for small datasets)

```sql
CREATE INDEX idx_tasks_tags ON tasks(tags) WHERE tags IS NOT NULL;
```

- Indexes the entire JSON array (not individual elements)
- Speeds up queries that check for tag existence
- Doesn't help with specific tag lookups

**Solution 2: Expression Index** (SQLite 3.9.0+)

```sql
-- Index for specific tag (e.g., "frontend")
CREATE INDEX idx_tasks_tag_frontend ON tasks(
    (SELECT 1 FROM json_each(tasks.tags) WHERE json_each.value = 'frontend')
) WHERE tags IS NOT NULL;
```

- Indexes a specific tag value
- Very fast for that specific tag
- Requires one index per commonly queried tag
- Not practical for dynamic tag sets

**Solution 3: Materialized View** (for large datasets)

```sql
-- Create a materialized view of task-tag pairs
CREATE TABLE task_tags_materialized (
    task_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (task_id, tag)
);

CREATE INDEX idx_task_tags_mat_tag ON task_tags_materialized(tag);

-- Trigger to keep materialized view in sync
CREATE TRIGGER sync_task_tags_insert AFTER INSERT ON tasks
WHEN NEW.tags IS NOT NULL
BEGIN
    INSERT INTO task_tags_materialized (task_id, tag)
    SELECT NEW.id, json_each.value
    FROM json_each(NEW.tags);
END;

CREATE TRIGGER sync_task_tags_update AFTER UPDATE ON tasks
WHEN NEW.tags IS NOT NULL
BEGIN
    DELETE FROM task_tags_materialized WHERE task_id = NEW.id;
    INSERT INTO task_tags_materialized (task_id, tag)
    SELECT NEW.id, json_each.value
    FROM json_each(NEW.tags);
END;

CREATE TRIGGER sync_task_tags_delete AFTER DELETE ON tasks
BEGIN
    DELETE FROM task_tags_materialized WHERE task_id = OLD.id;
END;
```

- Denormalizes JSON array into junction table
- Very fast queries (standard SQL joins)
- Automatically synced via triggers
- More storage overhead

**Recommendation**:

- **Phase 1**: Use partial index (simple, adequate for < 10K tasks)
- **Phase 2**: If performance degrades, add materialized view

### 5.3 Performance Benchmarks

**Test Setup**:

- 10,000 tasks, each with 3-5 tags
- Query: Find tasks with "frontend" tag

**Results** (estimated, based on SQLite JSON performance):

- **Partial Index**: ~10-50ms (full table scan with JSON parsing)
- **Materialized View**: ~1-5ms (indexed lookup)

**Threshold**: If queries take > 100ms, consider materialized view.

### 5.4 Storage Overhead

**JSON Array Storage**:

- Empty array `[]`: 2 bytes
- Single tag `["frontend"]`: ~15 bytes
- Three tags `["frontend", "urgent", "refactor"]`: ~35 bytes

**Comparison**:

- **JSON Array**: ~35 bytes per task (3 tags)
- **Junction Table**: ~60 bytes per task (3 rows × 20 bytes)

**Conclusion**: JSON array is more space-efficient for typical tag counts (< 10 tags per item).

---

## 6. Tag Conventions & Best Practices

### 6.1 Recommended Tag Categories

**Functional Tags** (what the task does):

- `frontend`, `backend`, `database`, `api`, `cli`, `ui`, `ux`
- `testing`, `documentation`, `deployment`, `monitoring`
- `security`, `performance`, `accessibility`

**Status Tags** (task state):

- `blocked`, `urgent`, `high-priority`, `low-priority`
- `needs-review`, `needs-testing`, `ready-to-merge`
- `experimental`, `deprecated`, `legacy`

**Type Tags** (kind of work):

- `bugfix`, `feature`, `refactor`, `research`, `investigation`
- `cleanup`, `optimization`, `migration`, `upgrade`

**Domain Tags** (project area):

- `auth`, `payments`, `notifications`, `search`, `analytics`
- `admin`, `user-facing`, `internal-tools`

**Tech Stack Tags**:

- `rust`, `typescript`, `python`, `react`, `nextjs`
- `postgres`, `redis`, `docker`, `kubernetes`

### 6.2 Tag Naming Conventions

**Rules**:

1. **Lowercase**: All tags should be lowercase (e.g., `frontend`, not `Frontend`)
2. **Hyphenated**: Use hyphens for multi-word tags (e.g., `high-priority`, not `high_priority` or `highPriority`)
3. **Singular**: Use singular form (e.g., `bug`, not `bugs`)
4. **Concise**: Keep tags short (< 20 characters)
5. **Consistent**: Use the same tag for the same concept (e.g., don't mix `frontend` and `front-end`)

**Anti-Patterns**:

- ❌ `URGENT!!!` (use `urgent`)
- ❌ `needs_review` (use `needs-review`)
- ❌ `Front-End` (use `frontend`)
- ❌ `this-is-a-very-long-tag-name` (too long, be concise)

### 6.3 Tag Validation

**Application-Layer Validation** (in Rust):

```rust
pub fn validate_tag(tag: &str) -> Result<(), String> {
    if tag.is_empty() {
        return Err("Tag cannot be empty".to_string());
    }
    if tag.len() > 30 {
        return Err("Tag too long (max 30 characters)".to_string());
    }
    if !tag.chars().all(|c| c.is_ascii_lowercase() || c == '-') {
        return Err("Tag must be lowercase with hyphens only".to_string());
    }
    Ok(())
}

pub fn validate_tags(tags: &[String]) -> Result<(), String> {
    for tag in tags {
        validate_tag(tag)?;
    }
    if tags.len() > 10 {
        return Err("Too many tags (max 10)".to_string());
    }
    Ok(())
}
```

**Usage in MCP Tools**:

```rust
#[tool(description = "Create a new task")]
async fn task(action: "create")(&self, params: Parameters<TaskCreateRequest>) -> Result<CallToolResult, McpError> {
    let req = params.into_inner();
    
    // Validate tags
    if let Some(ref tags) = req.tags {
        validate_tags(tags).map_err(|e| McpError::invalid_params(e, None))?;
    }
    
    // Create task...
}
```

### 6.4 Tag Autocomplete & Suggestions

**Strategy**: Suggest tags based on:

1. **Existing Tags**: Most frequently used tags
2. **Context**: Tags from similar tasks (same mode, priority, etc.)
3. **Description**: Extract keywords from task description

**Implementation**:

```rust
pub fn suggest_tags(&self, description: &str, mode: &str, limit: usize) -> Result<Vec<String>> {
    // 1. Get most frequent tags
    let frequent_tags = self.get_task_tag_counts()?;
    
    // 2. Get tags from similar tasks (same mode)
    let similar_task_tags = self.get_tags_by_mode(mode)?;
    
    // 3. Extract keywords from description (simple heuristic)
    let keywords = extract_keywords(description);
    
    // 4. Combine and rank suggestions
    let mut suggestions = Vec::new();
    // ... ranking logic ...
    
    suggestions.truncate(limit);
    Ok(suggestions)
}
```

---

## 7. Use Cases & Examples

### 7.1 Filter High-Priority Frontend Tasks

**MCP Call**:

```json
{
  "tool": "task(action: "list")",
  "params": {
    "priority": "P1",
    "tags": ["frontend"],
    "tag_mode": "any"
  }
}
```

**SQL**:

```sql
SELECT * FROM tasks 
WHERE priority = 'P1' 
  AND EXISTS (
    SELECT 1 FROM json_each(tasks.tags) 
    WHERE json_each.value = 'frontend'
  );
```

### 7.2 Find All Blocked Tasks

**MCP Call**:

```json
{
  "tool": "task(action: "list")",
  "params": {
    "tags": ["blocked"]
  }
}
```

**SQL**:

```sql
SELECT * FROM tasks 
WHERE EXISTS (
  SELECT 1 FROM json_each(tasks.tags) 
  WHERE json_each.value = 'blocked'
);
```

### 7.3 Find Security-Related Tech Debt

**MCP Call**:

```json
{
  "tool": "debt(action: "list")",
  "params": {
    "tags": ["security"],
    "status": "open"
  }
}
```

**SQL**:

```sql
SELECT * FROM tech_debt 
WHERE status = 'open' 
  AND EXISTS (
    SELECT 1 FROM json_each(tech_debt.tags) 
    WHERE json_each.value = 'security'
  );
```

### 7.4 List All Tags with Counts

**MCP Call**:

```json
{
  "tool": "tag_list",
  "params": {
    "entity": "all",
    "include_counts": true
  }
}
```

**Response**:

```json
{
  "tags": [
    {"tag": "frontend", "count": 42, "entities": ["tasks"]},
    {"tag": "backend", "count": 38, "entities": ["tasks"]},
    {"tag": "security", "count": 15, "entities": ["tasks", "tech_debt"]},
    {"tag": "database", "count": 12, "entities": ["tech_debt"]},
    ...
  ]
}
```

### 7.5 Autocomplete Tags

**MCP Call**:

```json
{
  "tool": "tag_list",
  "params": {
    "prefix": "fro",
    "limit": 5
  }
}
```

**Response**:

```json
{
  "tags": [
    {"tag": "frontend"},
    {"tag": "front-end"}  // If it exists (should be normalized to "frontend")
  ]
}
```

### 7.6 Create Task with Tags

**MCP Call**:

```json
{
  "tool": "task(action: "create")",
  "params": {
    "description": "Implement user authentication",
    "priority": "P1",
    "mode": "fixer",
    "tags": ["backend", "security", "auth", "urgent"]
  }
}
```

**Result**: Task created with tags `["backend", "security", "auth", "urgent"]`.

---

## 8. Implementation Roadmap

### Phase 1: Schema & Core Queries (Week 1)

**Goals**:

1. Add `tags` column to `tasks` and `tech_debt` tables (migration 003)
2. Implement core query functions in `db.rs`:
   - `query_tasks_by_tag()`
   - `query_tasks_by_tags_or()`
   - `query_tasks_by_tags_and()`
   - `list_all_task_tags()`
3. Update `DbTask` and `DbTechDebt` structs to include `tags` field
4. Write unit tests for tag queries

**Deliverables**:

- `.agent/migrations/003_add_tags.sql`
- Updated `db.rs` with tag query functions
- Unit tests for tag queries

### Phase 2: MCP Tool Extensions (Week 2)

**Goals**:

1. Extend `task(action: "list")` and `debt(action: "list")` tools with tag filtering
2. Implement new `tag_list` tool
3. Extend `task(action: "create")`, `task(action: "update")`, `debt(action: "add")` tools with tag support
4. Add tag validation logic
5. Write integration tests for MCP tools

**Deliverables**:

- Updated `server_tools/tasks.rs` and `server_tools/debt.rs`
- New `server_tools/tags.rs` module
- Integration tests for tag filtering

### Phase 3: Default Tags & Migration (Week 3)

**Goals**:

1. Implement default tag population logic
2. Run migration on existing database
3. Verify tag data integrity
4. Document tag conventions in `.agent/README.md`
5. Create tag usage guide

**Deliverables**:

- `db.rs::populate_default_tags()`
- Updated `.agent/README.md` with tag conventions
- Tag usage guide: `.agent/docs/tags.md`

### Phase 4: Advanced Features (Week 4)

**Goals**:

1. Implement tag autocomplete
2. Implement tag suggestions (based on description, mode, etc.)
3. Add tag frequency/cloud API
4. Optimize queries with indexes (if needed)
5. Performance testing and benchmarking

**Deliverables**:

- `db.rs::autocomplete_task_tags()`
- `db.rs::suggest_tags()`
- `db.rs::get_task_tag_counts()`
- Performance benchmarks and optimization report

---

## 9. Success Metrics

### 9.1 Quantitative Metrics

1. **Query Performance**: Tag queries complete in < 50ms for 10K tasks
2. **Storage Overhead**: < 5% increase in database size
3. **Adoption Rate**: > 80% of new tasks/debt items have at least one tag
4. **Tag Diversity**: > 50 unique tags across all tasks/debt items

### 9.2 Qualitative Metrics

1. **User Satisfaction**: Users find tag filtering useful and intuitive
2. **Tag Quality**: Tags are consistent and follow naming conventions
3. **Discoverability**: Users can easily find related tasks via tags

### 9.3 Operational Metrics

1. **Tag Coverage**: % of tasks/debt items with tags (target: > 90%)
2. **Tag Consistency**: % of tags following naming conventions (target: > 95%)
3. **Query Usage**: % of `task(action: "list")`/`debt(action: "list")` calls using tag filters (target: > 30%)

---

## 10. Future Enhancements

### 10.1 Tag Hierarchies

**Idea**: Support hierarchical tags (e.g., `frontend/react`, `backend/api`).

**Benefits**:

- More granular categorization
- Easier to filter by broad category (e.g., all `frontend/*` tags)

**Implementation**:

- Store tags as paths (e.g., `"frontend/react"`)
- Query with prefix matching (e.g., `WHERE tag LIKE 'frontend/%'`)

### 10.2 Tag Aliases

**Idea**: Support tag aliases (e.g., `frontend` = `front-end` = `fe`).

**Benefits**:

- Handles typos and variations
- Easier migration from legacy tags

**Implementation**:

- Add `tag_aliases` table mapping canonical tag to aliases
- Normalize tags on insert/query

### 10.3 Tag Recommendations (ML-Based)

**Idea**: Use ML to suggest tags based on task description.

**Benefits**:

- Automatic tagging for new tasks
- Improved tag consistency

**Implementation**:

- Train classifier on existing task descriptions + tags
- Call model API to predict tags for new tasks

### 10.4 Tag-Based Workflows

**Idea**: Trigger workflows based on tag changes (e.g., task tagged "urgent" → notify orchestrator).

**Benefits**:

- Automated task routing
- Event-driven orchestration

**Implementation**:

- Add triggers on tag updates
- Dispatch events to orchestrator

### 10.5 Tag Analytics Dashboard

**Idea**: Web dashboard showing tag usage trends, tag clouds, tag co-occurrence.

**Benefits**:

- Visual insights into project structure
- Identify tag gaps or over-tagging

**Implementation**:

- Web server serving dashboard (React or similar)
- Query database for tag statistics
- Visualizations (charts, graphs, tag clouds)

---

## 11. References

- **SQLite JSON Functions**: <https://www.sqlite.org/json1.html>
- **Database Schema**: `.agent/schema.sql`
- **MCP Server**: `agent-infra-mcp/src/server.rs`
- **Task Database**: `agent-infra-mcp/src/db.rs`

---

## 12. Open Questions

1. **Tag Limit**: Should we enforce a maximum number of tags per item? (Recommendation: 10)
2. **Tag Normalization**: Should we auto-normalize tags (e.g., `Front-End` → `frontend`)? (Recommendation: Yes)
3. **Tag Deletion**: Should we allow deleting tags from all items? (Recommendation: Yes, via admin tool)
4. **Tag Merging**: Should we support merging tags (e.g., `front-end` → `frontend`)? (Recommendation: Phase 2)
5. **Tag Permissions**: Should some tags be restricted (e.g., only orchestrator can add "urgent")? (Recommendation: No, keep simple)

---

**End of Research Report**
