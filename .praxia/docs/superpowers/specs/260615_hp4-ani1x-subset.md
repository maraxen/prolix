---
name: hp4-ani1x-subset
description: HP4 ANI-1x 16-system curation criteria for §7.1 paper-gating figure (#260 → #259)
metadata:
  type: project
---

# HP4 ANI-1x Subset — Curation Criteria for §7.1

**Date:** 2026-06-15
**Task:** 260615_sprint39 (#328)
**Status:** approved
**Prior spec:** `docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md`
**Prior synthesis:** `.praxia/docs/2026-05-20-hp4-research-synthesis.md`
**Implementation:** `scripts/data/fetch_ani1x_subset.py` (current); `scripts/curate/hp4_ani1x_subset.py` (target)

This document specifies exactly how the 16-system ANI-1x subset for the §7.1 paper-gating figure is curated. It supersedes the selection-criteria and storage sections of the prior spec and incorporates all empirical corrections (R10–R13, MMFF94 pivot, COMP6v2 schema differences, force-unit correction). Narrative context and the complete risk register live in the prior spec.

---

## §1 Purpose and Paper Connection

The §7.1 paper-gating figure demonstrates Claim 1 (heterogeneous-batch substrate) × S1 (differentiability): fit bonded force-field parameters against DFT reference forces via `jax.grad` through `EnsemblePlan.run(n_steps=0).gradient()`.

**Novelty (research-confirmed, 39-source synthesis):** no prior work (espaloma, ForceBalance, chemtrain, JAX-MD, reversible-DMS) has reported wall-clock comparisons between batched-vmap and looped-per-system gradient evaluation on *heterogeneous-topology* ensembles. This claim requires a dataset that is genuinely topologically heterogeneous and spans multiple atom-count buckets.

---

## §2 Dataset Provenance

### 2.1 ANI-1x (primary — Lane A)

| Field | Value |
|---|---|
| Name | ANI-1x |
| Paper DOI | 10.1038/s41597-020-0473-z (Smith et al., Sci Data 2020) |
| Dataset DOI | 10.6084/m9.figshare.10047041.v1 |
| Format | Single HDF5 archive |
| Pinned SHA-256 | `fe0ba06198ee72cf1003deebab2652097f6ab518337784dc811fa7da0c3bf5ac` |
| Figshare MD5 | `98090dd6679106da861f52bed825ffb7` |
| QM level | ωB97X/6-31G* (Gaussian 09) |
| Elements | H, C, N, O only |
| Total conformers | 5,496,771 across 63,865 unique molecules |
| Atom-count range | Mean 15 total (8 heavy), σ=5; max 63 atoms |

**HDF5 schema (flat dot-notation keys, per-isomer grouping):**

| Key | Shape | Dtype | Units |
|---|---|---|---|
| `coordinates` | `(Nc, Na, 3)` | float32 | Å |
| `atomic_numbers` | `(Na,)` | uint8 | — |
| `wb97x_dz.energy` | `(Nc,)` | float64 | Ha |
| `wb97x_dz.forces` | `(Nc, Na, 3)` | float32 | **Ha/Å** |

**CRITICAL — force unit correction:**
Forces are stored in **Hartree per Ångstrom (Ha/Å)**, NOT Ha/Bohr.
Conversion to kcal/mol/Å is:

```
force_kcal_per_mol_per_angstrom = force_Ha_per_angstrom * 627.5094740631
```

Do NOT divide by 0.5291772109 (Bohr radius). Including the Bohr factor would halve all reference forces and invalidate the fitting target.

### 2.2 COMP6v2 (supplement — Lane B)

| Field | Value |
|---|---|
| Title | COMP6v2 Release |
| Dataset DOI | 10.5281/zenodo.10126157 |
| Concept DOI | 10.5281/zenodo.10126156 (version-independent; cite this) |
| Required file | `COMP6v2_wB97X-631Gd.tar.gz` (167 MB) |
| Inner HDF5 SHA-256 | `e7c3e3e5db9fb7a64d00f86fb6b843323fae9dac8736a56c5875ef38051c81d0` |
| QM level | ωB97X/6-31G* (same as ANI-1x) |

**COMP6v2 HDF5 schema differs from ANI-1x in 5 ways:**

| Aspect | ANI-1x | COMP6v2 |
|---|---|---|
| Top-level groups | by molecular formula | **by atom count (`"006"`, `"007"`, ..., `"312"`)** |
| Forces key | `wb97x_dz.forces` | `forces` (bare) |
| Energy key | `wb97x_dz.energy` (singular) | `energies` (plural) |
| Species | `atomic_numbers`, shape `(Na,)` | `species`, shape `(Nc, Na)` |
| Group semantics | one group = one molecule | one group = all molecules of size N |

**Trp-cage (1L2Y):** group `312`, shape `(128, 312, 3)`.
**Chignolin (1UAO):** group `138`.
Both verified by local download 2026-05-20.

---

## §3 ATOM_BUCKETS Ladder Change

The prior ladder `(256, 1024, 5000, 25000, 60000)` makes cross-bucket demonstration impossible. HP4 requires widening to:

```python
# src/prolix/types/bundles.py
ATOM_BUCKETS = (64, 128, 256, 1024, 5000, 25000, 60000)
```

Under this ladder:
- bucket 0: n_atoms ≤ 64 (most Lane A molecules)
- bucket 1: 64 < n_atoms ≤ 128 (larger Lane A + Lane B mid-size)
- bucket 2: 128 < n_atoms ≤ 256 (Chignolin 138 atoms)
- bucket 3: 256 < n_atoms ≤ 1024 (Trp-cage 312 atoms)

The 20–21-molecule ensemble spans 4 distinct bucket slots. HP3 coarsening tests must remain green under the new ladder.

---

## §4 Lane A Selection Criteria (16 molecules from ANI-1x)

### 4.1 Filter chain (applied in sequence)

| Step | Filter | Description | Notes |
|---|---|---|---|
| F1 | Element whitelist | `all(Z in {1,6,7,8} for Z in atomic_numbers)` | ANI-1x guarantee; defensive assertion |
| F2 | Atom-count window | `5 ≤ n_total_atoms ≤ 35` | Spec range; impl uses 15–30 (see §4.4) |
| F3 | Conformer floor | `n_conf ≥ 20` (with forces non-NaN) | Sufficient for gradient averaging |
| F4 | Force validity | `not np.any(np.isnan(forces))` | Defensive |
| F5 | Max-force cap | `np.max(np.abs(forces_kcal)) < 200.0` kcal/mol/Å | Rejects pathological strained geometries |

### 4.2 Diversity selection — AIMNet2 local-environment hashing

For each non-H atom a in molecule M, compute hash tuple:
```
hash(a) = (Z_a, n_H_connected(a), n_total_neighbors(a),
           tuple(sorted(Z_b for b in non_H_neighbors(a))))
```

Bond detection: `bond(i,j) iff ||r_i - r_j|| < (rad[Z_i] + rad[Z_j]) * 1.20`
with Cordero 2008 radii: H=0.31, C=0.76, N=0.71, O=0.66 Å.

**Algorithm SELECT_16 (seed=42):**
1. Load all molecules passing F1–F5 via `iter_data_buckets(['wb97x_dz.energy', 'wb97x_dz.forces'])`
2. For each molecule M: compute `env_hash_set(M)`
3. `global_hash_freq = Counter(all hashes across all molecules)`
4. `rarity_score(M) = mean(1.0 / global_hash_freq[h] for h in env_hash_set(M))`
5. Sort by rarity_score descending
6. Greedy pick: select M if `not env_hash_set(M).issubset(picked_hashes)`; update picked_hashes; stop at 16
7. If len(selected) < 16: relax F3 to `n_conf ≥ 10`, restart
8. Sort selected by n_total_atoms ascending, then by isomer key ascending

### 4.3 Post-selection quality gates

- **Bonded-type coverage:** `|bond_types| ≥ 20 AND |angle_types| ≥ 30` across the 16 molecules (MMFF94 atom-type pairs). If either fails: hybrid scoring `0.5 * rarity_score + 0.5 * marginal_bonded_type_gain`, re-select.
- **ECFP4 similarity:** max pairwise Tanimoto < 0.85 across all 16 pairs.
- **Atom-count diversity:** ≥ 3 distinct atom counts in the selected set.
- **Visual inspection:** SMILES spot-checked for chemical plausibility.

### 4.4 Implementation note

Current implementation uses `LANE_A_N_ATOMS_MIN = 15`, `LANE_A_N_ATOMS_MAX = 30`. The task-spec range 5–35 is authoritative; 15–30 is empirical. If post-selection quality gates pass with 15–30, no change needed. If bonded-type coverage fails (R9), widen to 5–35 and re-run SELECT_16.

---

## §5 Lane B Selection Criteria (4–5 molecules from COMP6v2)

Lane B is the **held-out test set**. Parameters are never updated against Lane B. The §7.1 primary metric is Lane B force RMSE after Lane A training converges.

### 5.1 Mandatory molecules

| Name | PDB ID | COMP6v2 group | n_atoms | Bucket |
|---|---|---|---|---|
| Trp-cage | 1L2Y | `"312"` | 312 | 3 (≤ 1024) |
| Chignolin | 1UAO | `"138"` | 138 | 2 (≤ 256) |

### 5.2 Required additional molecules (bucket 1)

From COMP6v2 ANI-MD or DrugBank subsets: 2 molecules with 60–128 atoms (bucket 1). Rank by atom count descending within bucket; take the two largest distinct structures.

### 5.3 Optional

1 Tripeptide from COMP6v2 Tripeptides subset, 50–128 atoms.

### 5.4 Total ensemble

16 Lane A + 4–5 Lane B = **20–21 molecules spanning 4 buckets (0–3)**.

---

## §6 Per-System Metadata Schema

### 6.1 Directory layout

```
data/ani1x_subset/
  manifest.json
  <mol_id>/            # mol_000 through mol_015 for Lane A
    coords.npy         # (n_conformers, n_atoms, 3) float32, Angstrom
    forces.npy         # (n_conformers, n_atoms, 3) float32, kcal/mol/Angstrom
    energy.npy         # (n_conformers,) float64, Hartree
    meta.json
    params_init.json
  trp_cage/ chignolin/ anim_drug_0/ drugbank_0/  # Lane B
```

### 6.2 meta.json schema

```json
{
  "mol_id": "mol_000", "lane": "a", "source": "ANI-1x",
  "source_doi": "10.6084/m9.figshare.10047041.v1",
  "isomer_key": "C7H8N2O1", "smiles": "...",
  "n_total_atoms": 18, "n_heavy_atoms": 9,
  "atomic_numbers": [6, 1, ...],
  "n_conformers": 47, "eq_conf_idx": 23,
  "bucket_idx": 0, "bucket_ladder": [64, 128, 256, 1024, 5000, 25000, 60000],
  "rarity_score": 0.0123, "mmff94_typed": true,
  "coords_sha256": "...", "forces_sha256": "...", "energy_sha256": "...",
  "force_unit_conversion": "Ha/Ang * 627.5094740631 = kcal/mol/Ang",
  "selection_seed": 42, "curation_date": "2026-06-15",
  "spec_version": "260615_hp4-ani1x-subset"
}
```

Lane B additions: `"lane": "b"`, `"source": "COMP6v2"`, `"source_doi": "10.5281/zenodo.10126157"`, `"comp6v2_group": "312"`, `"pdb_id": "1L2Y"`.

### 6.3 params_init.json schema

MMFF94-derived bonded parameters. Notes:
- `k_bond`: uniform 400 kcal/mol/Å² (starting value; gradient descent updates)
- `k_angle`: uniform 50 kcal/mol/rad²
- `r0` and `theta0_deg`: from **minimum-energy conformer** (`eq_conf_idx = argmin(energy.npy)`) — R13 fix (first conformers may be strained with C-C bonds up to 2.28 Å)
- `k_phi`: initialized to 0.0; `periodicity=[3]`, `phase_deg=[0.0]` — one Fourier term per torsion (R11 fix; v0 emitted empty arrays giving zero learnable DOF)

---

## §7 Differentiable Fitting Loss (for §7.1 implementation)

```
L_m = (1/N_atoms_m) * mean[(F_pred - F_ref)^2]_{atoms,conf}
       + alpha * (1/N_conf_m) * mean[(E_pred - E_ref - shift_m)^2]_{conf}
       + w_reg * sum[(theta - theta_init)^2 / sigma_theta^2]_{params}

L_total = sum_m L_m  (Lane A only; Lane B is held-out eval)
```

| Parameter | Value | Source |
|---|---|---|
| `alpha` (energy weight) | 0.25 | TorchANI Table 3 sweep |
| `shift_m` | `mean(E_ref) - mean(E_pred)`, recomputed each step | espaloma SI |
| `w_reg` | 0.01 | ForceBalance defaults |
| `sigma_bond_k` | 100 kcal/mol/Å² | harmonic prior |
| `sigma_bond_r0` | 0.05 Å | harmonic prior |
| `sigma_angle_k` | 30 kcal/mol/rad² | harmonic prior |
| `sigma_angle_theta0` | 5° (0.0873 rad) | harmonic prior |

**Primary metric:** Lane B force RMSE after training converges.

---

## §8 MMFF94 Parameterization Toolchain

**Why RDKit MMFF94 (not OpenFF Toolkit):** OpenFF Toolkit is unavailable on PyPI (yanked 0.18.0; conda-forge dep resolve is OOM-intensive). RDKit MMFF94 is a project dependency, pip-installable, and sufficient for initial parameter guesses.

**Bond perception:** `rdDetermineBonds.DetermineBonds(mol, charge=0)` (fallback: `DetermineConnectivity`)

**Atom typing:** `AllChem.MMFFGetMoleculeProperties(mol).GetMMFFAtomType(i)`

**Equilibrium geometry source:** `argmin(energy.npy)` (R13 fix).

---

## §9 Exit Criteria

| # | Criterion | Verification |
|---|---|---|
| 1 | `sha256(ani1x.h5)` matches pinned constant | Script exit-code 0 |
| 2 | `sha256(comp6v2.h5)` matches pinned constant | Script exit-code 0 |
| 3 | `data/ani1x_subset/` has exactly 16 Lane A + ≥4 Lane B dirs | `ls` count |
| 4 | `manifest.json` regenerates hash-identically (seed=42) | Re-run script, diff manifests |
| 5 | All Lane A: n_conformers ≥ 20 with non-NaN forces | meta.json inspection |
| 6 | All species ∈ {1, 6, 7, 8} | meta.json `atomic_numbers` |
| 7 | Force unit conversion verified: `forces.npy * (1/627.5094740631)` in [0, ~0.5] Ha/Å | Spot-check 3 mols |
| 8 | SHA-256 of each numpy array matches meta.json after rsync | Compare pre/post |
| 9 | Cluster L3 gate: Trp-cage `EnsemblePlan(B=1).run(n_steps=0).gradient()` returns non-NaN | SLURM exit-code 0 |
| 10 | Bucket span: Lane B contains ≥3 molecules with `bucket_idx ≥ 1` | manifest.json inspection |
| 11 | Bonded-type coverage: `n_bond_types ≥ 20 AND n_angle_types ≥ 30` across Lane A | manifest.json |
| 12 | ECFP4 pairwise Tanimoto < 0.85 across all Lane A pairs | Post-selection gate |
| 13 | All 20–21 molecules parameterize via MMFF94; `params_init.json` written for all | Script log |
| 14 | Trp-cage: `params_init.json` has ≥ 300 bonds, ≥ 500 angles, ≥ 800 proper torsions | Manual check (empirical: 318, 576, 857) |
| 15 | Lane B held-out: training script refuses Lane B parameter updates | CI test |
| 16 | `baseline_loop_wallclock_s` recorded; if < 1.0 s → escalate ensemble size | manifest.json |
| 17 | `ATOM_BUCKETS=(64,128,256,1024,5000,25000,60000)` in `bundles.py`; HP3 tests green | CI |

---

## §10 Key Risks

| Risk | Status | Resolution |
|---|---|---|
| R4 — force unit error (Ha/Bohr vs Ha/Å) | CLOSED | Multiplier 627.5094740631 only; no Bohr factor |
| R10 — Lane attribution bug | CLOSED | Lane derived from path, not HDF5 key |
| R11 — Empty torsion arrays | CLOSED | One Fourier term per torsion (periodicity=[3]) |
| R12 — MMFF-untypable molecules | CLOSED | Min-energy conformer (R13 fix) resolves typing |
| R13 — First-conformer r0/theta0 bias | CLOSED | Use argmin(energy.npy) for equilibrium geometry |
| R5 — AIMNet2 diversity collapse | OPEN | Mitigated by bonded-type coverage gate (exit criterion 11) |
| R6 — Baseline wall-clock floor too low | OPEN | Must record and check; escalate to 64 mols if < 1.0 s |
| R7 — Bonded-param toolchain | CLOSED | MMFF94 pivot; 20/20 parameterize |
| R8 — Train/test split not enforced | CLOSED | Lane B held-out; CI test required (exit criterion 15) |

Full risk details: `docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md` §9.

---

## §11 Citations

```bibtex
@article{smith2020ani1x,
  title={The {ANI}-1ccx and {ANI}-1x data sets},
  author={Smith, Justin S. and others},
  journal={Scientific Data}, volume={7}, pages={134}, year={2020},
  doi={10.1038/s41597-020-0473-z}
}

@dataset{comp6v2_2023,
  title={{COMP6v2 Release}},
  author={Smith, Justin S. and others},
  year={2023}, publisher={Zenodo},
  doi={10.5281/zenodo.10126157}
}

@article{anstine2025aimnet2,
  title={{AIMNet2}: A neural network potential},
  author={Anstine, Dylan M. and others},
  journal={Chemical Science}, year={2025},
  doi={10.1039/d4sc08572h}
}
```
