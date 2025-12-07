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

- Improve modularity and ensure complete scoping of ForceField class
- Support lazy loading
- Add validation schemas
- Add support for GAFF/GAFF2
- Add support for SMIRNOFF
- Add support for ligands
- Add support for polarizable water
- Add support for virtual sites
- Add support for periodic boundary conditions
- Add support for explicit solvation
- Add support for PME
- Add support for GBSA/OBC2

### 3. Code Optimization

**Scope:** `src/prolix/physics/`

- Profile JIT compilation times
- Pre-compute CMAP coefficients (avoid per-step spline fitting)
- Consider `jax.lax.scan` for neighbor list updates, vectorization, or jax-md neighbor list updates where
available

### 4. Simulation Loop Specification

**Scope:** New `src/prolix/simulate.py`

```python
@dataclass
class SimulationSpec:
    total_time_ns: float
    step_size_fs: float = 2.0
    save_interval_ns: float = 0.001  # 1 ps
    accumulate_steps: int = 500      # inner scan loop
    save_path: str = "trajectory.array_record"

```

```python
class SimulationState(eqx.Module):
    ...

    def to_array_record(self):
        packed_states = m.packb(self.numpy())
        return writer.write(packed_states)
        
    
    @classmethod
    def from_array_record(cls, packed_states):
        return cls(**m.unpackb(packed_states))
    
    def numpy(self):
        return jax.tree_util.tree_map(np.asarray, self)

```

```python
import msgpack_numpy as m
m.patch()
from array_record.python import ArrayRecordWriter
writer = ArrayRecordWriter(...)
def write_trajectory(states):
    # convert to msgpack bytes
    msgpack_bytes = m.packb(states)
    # write to file
    writer.write(msgpack_bytes)

for epoch in range(total_epochs):  # outer Python loop (dynamic)
    def wrap_step(state,):
        return jax.fori_loop(0, int(total_time_ns / save_interval_ns), step_fn, state)
    
    accumulated_states = jax.lax.scan(wrap_step, state, None, length=accumulate_steps)
    jax.block_until_ready(accumulated_states)
    state = accumulated_states[-1]
    cpu_states = jax.device_put(accumulated_states, 'cpu')
    write_trajectory(cpu_states)

```

- analyze chingnolin trajectory and compare to openmm
    1. calculate RMSD
    2. calculate free energy
    3. calculate binding free energy
    4. calculate dihedral distributions and validity over the simulation
    5. calculate contact map

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
- RDKit integration

### 7. Periodic Boundary Conditions and Explicit Solvation

**Scope:** `src/prolix/physics/`

- Implement periodic boundary conditions
- Implement PME (NOTE: jax-md.energy may or may not have this already. be sure to check the actual source files jax_md/src/python/.../_energy.py)
- Implement explicit solvation
- benchmark performance with periodic boundary conditions against OpenMM

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
