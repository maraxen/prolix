# Pre-reg research notes — what we need to measure to finalize thresholds

Per `using-bathos` "Research Before Finalizing Pre-Registration": the sidecar's
thresholds must be **anchored in evidence**, not guessed. This doc tracks what
research is still needed before the campaign can run for real.

## Anchoring questions (all need empirical answers)

1. **What is the prolix v1.2 Bundle-code per-mol-step wall-clock at N=64 on RTX Pro 6000?**
   - Current evidence: pre-Bundle code measured ~14.5s / 500 steps for N=64 batched on the same hardware (sweep cell, 2026-05-20).
   - Bundle-code may differ. Need fresh single-cell smoke.
   - Action: submit `bench_prolix.py --n-mols 64 --n-steps 500` to cluster (single cell, ~1 min).

2. **What is DMFF's per-mol-step on the same primitive?**
   - Source-verified 2026-05-22 from `/tmp/dmff-repo` (commit 809f656):
   - `Hamiltonian(*xml_files).createPotential(topo, nonbondedMethod=NoCutoff)` → one call per topology.
   - `getPotentialFunc()` returns `efunc(pos, box, pairs, prms)`. `pairs` must come from
     `NoCutoffNeighborList(cov_map=pot.meta['cov_map']).allocate(positions)` — shape `(N,3)`, third
     column is bond order. DO NOT pass a bare 2-col pairs array.
   - `getParameters()` returns a ParamSet pytree: `params['HarmonicBondForce']['k']` and `['length']`
     (in kJ/mol/nm² and nm). Angle key is `['HarmonicAngleForce']['k']` and `['angle']` (radians).
   - Batching: NO native multi-topology API. Build `loss = sum_i efunc_i(pos_i, box, pairs_i, params)`,
     then `jax.value_and_grad(loss)(params)` + optax step.
   - Unit conversions from prolix JSON (AKMA → OpenMM/DMFF):
       bonds:  length_nm = r0_Å * 0.1;  k_kJ = k_kcal * 4.184 * 100 * 2  (OpenMM half-spring)
       angles: angle_rad = theta0_deg * π/180;  k_kJ = k_theta_kcal * 4.184 * 2
       torsions: k_kJ = k_phi_kcal * 4.184
   - Input: OpenMM-compatible XML (HarmonicBondForce + HarmonicAngleForce + PeriodicTorsionForce).
     NonbondedForce is OPTIONAL — absent section prints a warning but does not throw.
   - Install: `uv sync --group benchmark-dmff --extra openmm` + `mamba install -c conda-forge freud`
   - Canonical example: `/tmp/dmff-repo/examples/classical/test_xml.py:54-85`
   - Action: write `bench_dmff.py` using real DMFF + `_generate_xml.py` utility.

3. **What is TorchMD's per-mol-step?**
   - Source-verified 2026-05-22 from `/tmp/torchmd-repo` (commit 0948418):
   - Correct package: `torchmd` (NOT `torchmd-net` which is NN potentials).
   - `Forces(parameters, terms=['bonds','angles','dihedrals'])` — strictly file-based.
     `Parameters` requires a `ForceField` object from PSF/YAML. Use `YamlForcefield(mol, yaml_path)`.
   - `Forces.compute(pos, box, forces)`: pos shape `(nreplicas, natoms, 3)`. Loops over systems at
     `forces.py:104-320` — one iteration per molecule. No batching over different topologies.
   - `torch.vmap` works only for multiple conformations of the SAME topology (same atom count).
   - YAML FF format (from `tests/water/water_forcefield.yaml`): keys `bonds: (AT1,AT2): {k0, req}`,
     `angles: (AT1,AT2,AT3): {k0, theta0}`. TorchMD bond k0 in kcal/mol/Å², req in Å — same units
     as prolix JSON 'k' and 'r0'. No unit conversion needed for bonds/angles.
   - Install: `uv sync --group benchmark-torchmd` + `uv pip install torch --index-url https://download.pytorch.org/whl/cu126`
   - Canonical example: `/tmp/torchmd-repo/tests/test_torchmd.py:396-409`
   - Action: write `bench_torchmd.py` — generate per-molecule YAML FF + minimal Molecule object.

4. **What is espaloma's per-mol-step on bonded-energy forward+backward?**
   - Source-verified 2026-05-22 from `/tmp/espaloma-repo` (commit 413eb55):
   - EnergyInGraph node feature keys (EXACT, from `mm/energy.py:18-19`):
       n2 (bonds):   'k' (force constant), 'eq' (equilibrium distance, Å)
       n3 (angles):  'k', 'eq' (radians)
       n4 (torsions): 'k' (barrier heights), 'phases', 'periodicity'
   - GeometryInGraph reads 'xyz' from n1 nodes, writes 'x' to n2/n3/n4 (`mm/geometry.py:165-242`).
   - Graph construction requires openff-toolkit (no toolkit-free path in repo). Use
     `Molecule.from_smiles(smiles)` → `relationship_indices_from_offmol(offmol)` → `dgl.heterograph`.
   - `dgl.batch` is NOT used anywhere in the espaloma codebase but DOES work on heterographs —
     use it for N-mol batching (this is maximally generous to espaloma).
   - Standalone energy path (from `mm/tests/test_energy.py:11-55`):
       `g.nodes['n1'].data['xyz'] = pos; geometry_in_graph(g); energy_in_graph(g, terms=['n2','n3','n4'])`
       energy in `g.nodes['g'].data['u']`, then `.backward()`.
   - Install: `uv sync --group benchmark-espaloma` + `mamba install -c dglteam/label/cu124 dgl -c conda-forge openff-toolkit`
   - Canonical example: `/tmp/espaloma-repo/espaloma/mm/tests/test_recoverability.py:140-178`
   - Action: write `bench_espaloma.py` — build DGL heterograph from JSON indices via openff-toolkit.

5. **Does ForceBalance even fit a 500-step Adam-equivalent comparison?**
   - ForceBalance uses Newton-Raphson + FD Hessian; converges in ~10 iterations.
   - Each iteration cost is much higher than an Adam step.
   - Action: skip Scope A (no autograd). For Scope C: measure wall-clock to converge to a fixed residual on a 16-mol target. Different methodology, report as ecological floor.

## Unified input: OpenMM XML

All harnesses accept OpenMM XML as a common input format. This ensures every tool receives
identical force-field parameters — no tool-specific param-conversion bugs can bias the results.

Architecture:
1. `_generate_xml.py` (shared utility in this directory): reads prolix JSON → writes one
   OpenMM-compatible `<ForceField>` XML per molecule. Sections: HarmonicBondForce +
   HarmonicAngleForce + PeriodicTorsionForce (no NonbondedForce). Unit conversions applied here.
2. `bench_prolix.py`: extended to accept `--xml-dir` instead of (or alongside) `--subset-dir`.
   Parses the XML and converts back to prolix internal representation. This makes prolix's input
   path identical to DMFF's — no special-casing for any tool.
3. `bench_dmff.py`: feeds the XML directly to `Hamiltonian(*[xml_paths]).createPotential(topo)`.
4. `bench_torchmd.py`: XML → YAML conversion (TorchMD uses YAML). Utility converts OpenMM XML
   to TorchMD YAML format so param provenance is identical.
5. `bench_espaloma.py`: XML → DGL heterograph. Uses openff-toolkit to build the graph, then
   injects k/eq from the XML (not from openff-toolkit's own SMIRNOFF FF).

XML generation per prolix JSON (unit conversions, from research 2026-05-22):
  bonds:   length_nm = r0_Å * 0.1;  k_kJ_nm2 = k_kcal_A2 * 4.184 * 100 * 2
  angles:  angle_rad = theta0_deg * π/180;  k_kJ_rad2 = k_theta_kcal_rad2 * 4.184 * 2
  torsions: k_kJ = k_phi_kcal * 4.184;  phase_rad = phase_deg * π/180

## Smoke plan (~1 day of work)

| Step | Action | ETA |
|---|---|---|
| 1 | Set up per-tool venvs in `scripts/benchmarks/external_baseline/.venv-*` | 1h |
| 2 | Write `bench_prolix.py` (copy/adapt fit_bonded_hp4.py logic) | 30min |
| 3 | Write `bench_dmff.py` (single-mol Adam loop using DMFF Hamiltonian) | 2h |
| 4 | Write `bench_torchmd.py` (per-system Adam loop) | 2h |
| 5 | Write `bench_espaloma.py` (drive `energy_in_graph` + per-mol Adam) | 2h |
| 6 | Smoke each at N=16 locally (CPU, just confirms it runs) | 1h |
| 7 | Cluster smoke at N=64 (one cell per tool, pinned node4007/4008) | 30min wait |
| 8 | Anchor thresholds in sidecar based on N=64 numbers | 30min |

After step 8: full campaign sweep at N ∈ {16, 32, 64, 128, 256, 512} × {f32, f64}
for the autograd-capable tools (4 tools × 12 cells = 48 cells per Scope A).

## Threshold-anchoring math (placeholder)

Once smoke numbers land, plug into:

```
prolix_n64 = T_p  # measured: 14.9 µs/mol-step (run 6ce95e65, job 14263023)
dmff_n64 = T_d    # TODO: smoke bench_dmff.py
torchmd_n64 = T_t # TODO: smoke bench_torchmd.py
espaloma_n64 = T_e # TODO: smoke bench_espaloma.py

# Conservative pass threshold (50% better than fastest non-batched competitor)
# PyTorch-from-scratch removed (2026-05-22): not a citable comparator; TorchMD covers PyTorch slot
pass_threshold = 0.5 * min(T_d, T_t)  # exclude espaloma (it has DGL batching)

# Marginal: between 0.5× and 1.0×
marginal_threshold = 1.0 * min(T_d, T_t)
```

If actual prolix < pass_threshold → claim succeeds.
If pass_threshold ≤ prolix < marginal_threshold → claim narrows to "competitive".
If prolix ≥ marginal_threshold → claim collapses (Scope B fallback).

## Hardware pin (mandatory)

All cluster submissions:
```
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4007,node4008
#SBATCH --gres=gpu:1
```

Pin to RTX Pro 6000 Blackwell only. Exclude node4009 (H200) — heterogeneous GPU
would confound the comparison.

## Pre-reg validity contract

This campaign **CANNOT BEGIN** (no `bth run` against the sidecar) until:
- [ ] All comparator harnesses smoke successfully at N=16 locally
- [ ] All comparator harnesses smoke successfully at N=64 on cluster
- [ ] Thresholds in sidecar are replaced with anchored values
- [ ] The sidecar's `outcomes.pass.reasoning` cites the smoke run's git_hash
- [ ] Sidecar is re-reviewed (by oracle if novel territory, otherwise self-check)

Until those gates pass, the sidecar's `outcomes.pass.condition` uses the
conservative bias-toward-not-falsely-claiming-success: if any comparator's smoke
fails to produce a valid number, the pass condition cannot be evaluated and the
residual `fail` branch fires.
