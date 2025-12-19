# Agent Handoff: Proxide Migration Execution

## Context

You are continuing a migration task in the `prolix` project. The user wants to replace the local `priox` folder with the authoritative `proxide` repository from GitHub, which will be added as a Git submodule.

## Repository Information

- **Workspace:** `/home/marielle/workspace/prolix`
- **Proxide Source:** `https://github.com/maraxen/proxide.git`
- **Current State:** `priox/` is a local folder with 242 files that needs to be replaced

## Your Mission

Execute the Proxide Migration Plan located at `.agent/PROXIDE_MIGRATION_PLAN.md`. Follow the phases in order:

### Phase 1: Repository Cleanup

1. Remove the existing `priox/` folder
2. Add `proxide` as a git submodule: `git submodule add https://github.com/maraxen/proxide.git proxide`

### Phase 2: Configuration Updates

1. Update `pyproject.toml`:
   - Replace `"priox"` with `"proxide"` in dependencies
   - Replace `"priox_rs"` with the appropriate Rust extension name
   - Update `[tool.uv.sources]` to point to `proxide` and `proxide/oxidize`
2. Check `proxide/pyproject.toml` for the actual package names

### Phase 3: Code Reference Updates

1. Search for all files with `priox` references using grep
2. Replace imports systematically:
   - `from priox.` → `from proxide.`
   - `import priox` → `import proxide`
   - Update notebook git clone URLs
3. Update all path references in scripts

### Phase 4: Rust Extension Integration

1. Verify the Rust extension structure in `proxide/oxidize/`
2. Build with `maturin develop --release`
3. Verify import works

### Phase 5: Agent Coordination Updates

1. Update `.agent/README.md` with new file paths
2. Update `.agent/HANDOFF.md` with new directory structure
3. Review `proxide/.agents/` for additional context

### Phase 6: Local Path Cleanup

1. Remove `src/priox.physics.force_fields/` if orphaned
2. Update any remaining path references

### Phase 7: Validation

1. Run `uv sync --reinstall`
2. Verify Python imports work
3. Run `pytest tests/` in both proxide and prolix
4. If tests fail, determine whether:
   - prolix needs to adapt to proxide (default)
   - proxide has an inaccuracy (document in `.agent/PROXIDE_DISCREPANCIES.md`)

### Phase 8: Git Commit

1. Stage all changes
2. Commit with the message template provided in the plan

## Ground Truth Priority

**IMPORTANT:** `proxide` is the authoritative source of truth.

- When equivalence tests fail, assume `prolix` needs to change
- If proxide has a bug/inaccuracy, document it in `.agent/PROXIDE_DISCREPANCIES.md`
- Do NOT modify proxide code without explicit user permission

## Files You Will Modify

- `pyproject.toml`
- `docs/source/*.md`
- `scripts/*.py`
- `notebooks/*.ipynb`
- `benchmarks/*.py`
- `tests/**/*.py`
- `src/prolix/**/*.py`
- `.agent/README.md`
- `.agent/HANDOFF.md`

## Success Criteria

- [ ] `uv sync` completes without errors
- [ ] `import proxide` works
- [ ] `import proxide_rs` (or `oxidize`) works
- [ ] All prolix tests pass
- [ ] No remaining references to `priox` in active code
- [ ] Git submodule properly tracks proxide repository

## Additional Notes

- The proxide repository has its Rust extension in `oxidize/` (not `rust_ext/`)
- The proxide repository has its own `.agents/` directory for agent coordination
- Some imports may still use `priox` as the package name internally - check proxide's pyproject.toml

Start by reading the full migration plan at `.agent/PROXIDE_MIGRATION_PLAN.md`, then execute each phase methodically.
