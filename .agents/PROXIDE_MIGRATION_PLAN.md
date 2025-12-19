# Proxide Migration Plan (Updated)

## Executive Summary

**Objective:** Replace the local `priox` folder with a Git submodule clone of [maraxen/proxide](https://github.com/maraxen/proxide), update all references, and establish `proxide` as the authoritative ground truth for protein I/O and Rust extensions.

**Priority:** High | **Estimated Effort:** 2-3 hours

---

## Proxide Repository Structure (Current - After Pull)

```
proxide/
├── .agents/              # Agent coordination docs
├── oxidize/              # Rust extension (handles parsing, force fields, MD)
│   ├── Cargo.toml        # name = "oxidize"
│   └── src/
├── src/
│   └── proxide/          # Python package (import as: from proxide import ...)
│       ├── assets/       # Force field XML files (source of truth)
│       │   ├── amber/
│       │   ├── charmm/
│       │   ├── gaff/
│       │   ├── implicit/
│       │   ├── openmm_bundled/
│       │   ├── water/
│       │   ├── protein.ff14SB.xml
│       │   └── protein.ff19SB.xml
│       ├── chem/
│       ├── core/
│       ├── geometry/
│       ├── io/
│       ├── md/
│       ├── ops/
│       └── physics/
│           └── force_fields/
│               ├── loader.py     # Python loader (uses oxidize.load_forcefield)
│               └── components.py
├── tests/
└── pyproject.toml        # name = "proxide", module-name = "oxidize"
```

**Key Changes from Original Plan:**

- ❌ **NO .eqx files** - Serialized force fields are eliminated
- ✅ **Force fields loaded from XML** via `oxidize.load_forcefield()`
- ✅ **Assets directory** contains all XML force field files

**Import Names:**

- **Python package:** `proxide` (import as `from proxide import ...`)
- **Rust extension:** `oxidize` (import as `import oxidize`)

---

## Phase 1: Repository Cleanup ✅ COMPLETE

```bash
rm -rf priox/
git submodule add https://github.com/maraxen/proxide.git proxide
git submodule update --init --recursive
```

---

## Phase 2: Configuration Updates ✅ COMPLETE

Updated `pyproject.toml`:

```toml
dependencies = [
    ...
    "proxide",
    "oxidize",
    ...
]

[tool.uv.sources]
proxide = { path = "proxide", editable = true }
oxidize = { path = "proxide/oxidize", editable = true }
```

---

## Phase 3: Code Reference Updates

### 3.1 Import Replacements ✅ DONE

| Old Pattern | New Pattern |
|-------------|-------------|
| `from priox.` | `from proxide.` |
| `import priox` | `import proxide` |
| `import priox_rs` | `import oxidize` |
| `priox_rs.` | `oxidize.` |

### 3.2 Force Field Loading Changes ⚠️ CRITICAL

**Old approach (eliminate):**

```python
# Loading serialized .eqx files - NO LONGER VALID
ff = force_fields.load_force_field("path/to/protein19SB.eqx")
```

**New approach:**

```python
# Load from XML via oxidize
import oxidize
ff_data = oxidize.load_forcefield("proxide/src/proxide/assets/protein.ff19SB.xml")
# OR use proxide's loader which wraps oxidize
from proxide.physics.force_fields import load_force_field
ff = load_force_field("protein.ff19SB")  # Looks up in assets
```

### 3.3 Path Replacements

| Old Path Pattern | Action |
|------------------|--------|
| `**/eqx/*.eqx` | Replace with XML path or use loader by name |
| `priox/src/priox/physics/force_fields/` | Use `proxide/src/proxide/assets/` |
| `sys.path.insert(...priox...)` | Remove - not needed with proper uv.sources |

### 3.4 Cleanup Commands

```bash
# Remove sys.path.insert lines for priox
find src tests scripts benchmarks -name "*.py" -exec sed -i '/sys.path.insert.*priox/d' {} \;

# Update remaining priox path references
find src tests scripts benchmarks -name "*.py" -exec sed -i 's|priox/src/priox|proxide/src/proxide|g' {} \;
find src tests scripts benchmarks -name "*.py" -exec sed -i 's|../priox/src|proxide/src|g' {} \;
```

---

## Phase 4: Rust Extension Integration

### 4.1 Build Rust Extension

```bash
cd proxide/oxidize
maturin develop --release
cd ../..
```

### 4.2 Verify Import

```python
import oxidize
print(oxidize.__file__)
oxidize.load_forcefield  # Should exist
```

---

## Phase 5: Agent Documentation Updates

### 5.1 Update `.agent/` docs

- Remove all references to `.eqx` files
- Update paths from `priox` to `proxide`
- Document new force field loading approach

---

## Phase 6: Validation

### 6.1 Build Verification

```bash
uv sync --reinstall
uv run python -c "import proxide; print(proxide.__file__)"
uv run python -c "import oxidize; print('Rust extension OK')"
```

### 6.2 Test Execution

```bash
uv run pytest tests/ -v --tb=short
```

### 6.3 Ground Truth Protocol

**`proxide` is the authoritative source.**

If tests fail:

1. **First:** Modify prolix to match proxide
2. **Exception:** If proxide has a documented inaccuracy, file an issue
3. **Document:** Any workarounds in `.agent/PROXIDE_DISCREPANCIES.md`

---

## Phase 7: Git Commit

```bash
git add -A
git commit -m "refactor: replace priox with proxide submodule

- Remove local priox folder
- Add proxide as git submodule from github.com/maraxen/proxide
- Update all priox imports to proxide
- Update priox_rs imports to oxidize
- Update force field loading to use XML via oxidize (no more .eqx)
- Update uv.sources to point to proxide/oxidize

BREAKING CHANGE: All priox imports now proxide, priox_rs now oxidize"
```

---

## Success Criteria

- [ ] `uv sync` completes without errors
- [ ] `import proxide` works
- [ ] `import oxidize` works  
- [ ] All prolix tests pass
- [ ] No remaining references to `priox` or `priox_rs` in active code
- [ ] No references to `.eqx` files
- [ ] Git submodule properly tracks proxide repository
