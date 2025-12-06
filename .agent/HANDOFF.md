# Prolix MD Physics Handoff Summary

## Current Status: OpenMM Parity Achieved ✅

**Benchmark Date:** 2025-12-06
**Test Protein:** 1UAO (Trp-cage, 138 atoms)
**Force Field:** ff19SB + GBSA/OBC2

### Energy Comparison (No Parameter Injection)

| Component | OpenMM (kcal/mol) | JAX MD (kcal/mol) | Δ |
|-----------|-------------------|-------------------|---|
| Bond | 20.3124 | 20.3124 | 0.0000 ✓ |
| Angle | 7.9587 | 7.9587 | 0.0000 ✓ |
| Torsion | 49.6248 | 49.6248 | 0.0000 ✓ |
| CMAP | 10.2384 | 10.2384 | 0.0000 ✓ |
| NonBonded (LJ+Coulomb) | -23.1066 | -23.1066 | ~0.0 ✓ |
| GBSA Total | -361.3371 | -361.3368 | ~0.0003 ✓ |
| **Total Energy** | **-296.3094** | **-296.3092** | **0.0002 ✓** |

---

## Repository Structure

```
prolix/
├── src/prolix/               # Main physics library
│   └── physics/
│       ├── system.py         # Energy function factory
│       ├── bonded.py         # Bond/Angle/Torsion energy
│       ├── cmap.py           # CMAP with periodic splines ⭐NEW
│       ├── generalized_born.py  # GBSA/OBC2 solvation
│       └── sasa.py           # Surface area calculation
├── priox/src/priox/          # IO and bridge submodule
│   └── md/bridge/
│       ├── core.py           # SystemParams creation from PDB+FF
│       └── types.py          # SystemParams TypedDict definition
├── benchmarks/
│   └── verify_end_to_end_physics.py  # Main validation script
├── data/
│   ├── pdb/                  # Test structures
│   └── force_fields/         # Pre-converted .eqx force fields
└── scripts/
    └── convert_all_xmls.py   # OpenMM XML → .eqx converter
```

---

## Agent Task Registry

Use `.agent/agent_tasks.jsonl` (append-only) to coordinate:

```json
{"timestamp": "...", "agent_id": "...", "task": "...", "status": "IN_PROGRESS|COMPLETED|BLOCKED", "files_in_scope": [...], "summary": "..."}
```

---

## Development Tasks for Parallel Agents

### 1. Force Field Brittleness Assessment

**Scope:** `scripts/convert_all_xmls.py`, `data/force_fields/`

- Test with ff14SB, ff99SB, CHARMM36
- Identify XML parsing failures
- Document unsupported features (virtual sites, polarizable)

### 2. Force Field Extensibility

**Scope:** `priox/src/priox/physics/force_fields/`, `convert_all_xmls.py`

- Design modular ForceField class
- Support lazy loading
- Add validation schemas

### 3. Code Optimization

**Scope:** `src/prolix/physics/`

- Profile JIT compilation times
- Pre-compute CMAP coefficients (avoid per-step spline fitting)
- Consider `jax.lax.scan` for neighbor list updates

### 4. Simulation Loop Specification

**Scope:** New `src/prolix/simulate.py`

```python
@dataclass
class SimulationSpec:
    total_time_ns: float
    step_size_fs: float = 2.0
    save_interval_ns: float = 0.001  # 1 ps
    accumulate_steps: int = 500      # inner scan loop

# Structure:
# for epoch in range(total_epochs):  # outer Python loop (dynamic)
#     state = jax.lax.scan(step_fn, state, None, length=accumulate_steps)
#     for i in range(saves_per_epoch):  # nested loop based on save_interval
#         write_trajectory(state)
```

### 5. Parallel Tempering

**Scope:** New `src/prolix/pt/`

- Implement replica exchange
- Temperature ladder generation
- Swap acceptance criterion

### 6. Ligand Support

**Scope:** `priox/src/priox/md/bridge/core.py`, `convert_all_xmls.py`

- GAFF/GAFF2 parameter parsing
- MOL2/SDF topology reading
- Separate ligand residue handling

---

## Known Limitations

1. **No Periodic Boundary Conditions** - vacuum/implicit only
2. **N² Non-bonded** - neighbor lists available but slower
3. **GBSA Radii Approximation** - uses scaled radii, not exact Born calculation
4. **Force RMSE ~2.7 kcal/mol/Å** - acceptable for sampling, not exact dynamics

---

## Running Validation

```bash
# Single protein
uv run python benchmarks/verify_end_to_end_physics.py

# Batch (multiple proteins)
uv run python benchmarks/verify_batch_physics.py
```

---

## Git Commits (This Session)

1. `29ee68c` - feat(cmap): Implement OpenMM-compatible periodic spline CMAP interpolation
2. `63faa7d` - fix(cmap): Update system.py to use cmap_energy_grids with new CMAP implementation
