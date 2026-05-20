# HP4 — ANI-1x + COMP6 DFT-Forces Subset Curation

**Date:** 2026-05-20
**Status:** Approved sub-spec (post-NotebookLM research synthesis, 39 sources)
**Task ID:** 260520_hp4_ani1x
**Roadmap anchors:** `docs/superpowers/specs/2026-05-19-prolix-long-horizon-roadmap.md` §2.7 (HP4), §7.1 (paper-gating figure)
**Notebook:** [Prolix HP4 — ANI-1x Curation](https://notebooklm.google.com/notebook/301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76) (`301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76`)
**Citation format:** Each claim cites `[NB:<source_id>; <URL/DOI>]` where `NB:` is the notebook source ID.

---

## §1 Motivation

The §7.1 paper-gating figure demonstrates Claim 1 (heterogeneous-batch substrate) × S1 (differentiability) multiplicatively: fit bonded force-field parameters (`bond_k`, `bond_r0`, `angle_k`, `angle_θ0`, `torsion_*`) against DFT reference forces via `jax.grad` through `EnsemblePlan.run(n_steps=0).gradient()`. Hard paper-submit gate.

**Novelty (research-confirmed):** No prior work — espaloma, ForceBalance, chemtrain, JAX-MD, reversible-DMS — has reported wall-clock comparisons between batched-`vmap` and looped-per-system gradient evaluation on *heterogeneous-topology* ensembles. JAX-MD's documented `vmap` use is over *homogeneous* parameter sweeps (particle diameters, RNG keys), not over varied molecular topologies; espaloma reports batched parametrization throughput but no AD-gradient-throughput comparison [NB:bfb13e74; semantic-scholar eabf/espaloma SI] [NB:3449e5b3; JAX-MD NeurIPS 2020] [NB:7330dfce; arXiv 2412.04374]. This is itself a paper claim.

---

## §2 Dataset Source

**Primary:** ANI-1x (Smith et al., *Sci Data* 2020).

| Field | Value |
|---|---|
| **Name** | ANI-1x |
| **Paper DOI** | [10.1038/s41597-020-0473-z](https://doi.org/10.1038/s41597-020-0473-z) [NB:74c012cd] |
| **Dataset DOI (item)** | [10.6084/m9.figshare.10047041.v1](https://doi.org/10.6084/m9.figshare.10047041.v1) |
| **Dataset DOI (collection)** | [10.6084/m9.figshare.c.4712477.v1](https://doi.org/10.6084/m9.figshare.c.4712477.v1) [NB:74c012cd] |
| **File format** | Single HDF5 archive [NB:74c012cd] |
| **License** | Publicly accessible (not explicitly CC-licensed in the data record; treat as research-redistributable; cite Smith 2020 on every use) [NB:51e8af53; arXiv 2408.12058] |
| **Elements** | H, C, N, O — **no sulfur**; S whitelist from initial roadmap is dropped [NB:74c012cd] [NB:8e72758c] |
| **QM level** | ωB97X/6-31G* (Gaussian 09) [NB:74c012cd] |
| **Total conformers** | 5,496,771 across 63,865 unique molecules [NB:9045d3c5; openqdc.io] [NB:65aa232f; LANL Smith] |
| **Atom-count range** | Mean 15 total (8 heavy), σ=5; **max 63 atoms** [NB:c0ac4c69; qchem/dataset.py] |
| **Support code** | https://github.com/aiqm/ANI1x_datasets [NB:74c012cd] |

**Supplement (mandatory for cross-bucket §7.1 evidence):** COMP6 benchmark suite (Smith et al. 2018 LANL).

| Field | Value |
|---|---|
| **Repository** | https://github.com/isayev/COMP6 [NB:65aa232f] |
| **QM level** | **ωB97X/6-31G* (Gaussian 09)** — same as ANI-1x; zero reference-baseline mixing [NB:65aa232f] |
| **File format** | HDF5 with the same `wb97x_dz.forces` / `wb97x_dz.energy` keys [NB:74c012cd] [NB:65aa232f] |
| **Forces** | Energies + atomic forces available for all non-equilibrium conformations [NB:65aa232f] |
| **Subsets relevant to §7.1** | DrugBank (837 mols, 44 atoms mean ±20), Tripeptides (248 mols, 53 atoms mean ±7), ANI-MD (14 mols, 75 atoms mean, **max 312 = Trp-cage 1L2Y**) [NB:65aa232f] |

**Why COMP6 (not SPICE) as supplement:** SPICE uses ωB97M-D3(BJ)/def2-TZVPPD, a different functional + dispersion correction → reference-baseline mixing penalty if combined with ANI-1x [NB:73f75bd6; arXiv 2209.10702]. COMP6 uses the **exact same** ωB97X/6-31G* level and same HDF5 schema → seamless [NB:65aa232f]. SPICE is the right *replacement* if we pivot off ANI-1x entirely; COMP6 is the right *supplement* for cross-bucket evidence.

**SPICE as documented fallback** (if S-containing dipeptides demanded by reviewers):

| Field | Value |
|---|---|
| **Dipeptides subset** | 33,850 conformers, all 676 combinations of 20 amino acids (incl. protonation variants + CYS-CYS disulfide), 26-60 atoms [NB:73f75bd6] |
| **DOI / accession** | [10.5281/zenodo.7338495](https://doi.org/10.5281/zenodo.7338495) [NB:73f75bd6] [NB:337e8307] |
| **License** | **CC0 (public domain)** [NB:337e8307; github.com/openmm/spice-dataset] |
| **QM level** | ωB97M-D3(BJ)/def2-TZVPPD [NB:73f75bd6] |
| **Elements** | H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I (15+ elements) [NB:337e8307] |

**Citation (BibTeX):**

```bibtex
@article{smith2020ani1x,
  title   = {The {ANI}-1ccx and {ANI}-1x data sets, coupled-cluster and
             density functional theory properties for molecules},
  author  = {Smith, Justin S. and Zubatyuk, Roman and Nebgen, Benjamin T.
             and Lubbers, Nicholas and Barros, Kipton and Roitberg, Adrian E.
             and Isayev, Olexandr and Tretiak, Sergei},
  journal = {Scientific Data}, volume = {7}, pages = {134}, year = {2020},
  doi     = {10.1038/s41597-020-0473-z}
}

@article{smith2018comp6,
  title   = {Less is more: {S}ampling chemical space with active learning},
  author  = {Smith, Justin S. and Nebgen, Ben and Lubbers, Nicholas and
             Isayev, Olexandr and Roitberg, Adrian E.},
  journal = {The Journal of Chemical Physics}, volume = {148}, number = {24},
  pages   = {241733}, year = {2018}, doi = {10.1063/1.5023802}
}
```

---

## §3 HDF5 Schema (verified)

The ANI-1x HDF5 uses **flat dot-notation keys** (e.g., `wb97x_dz.forces`, NOT `wb97x_dz/forces`), grouped per-isomer at the top level [NB:74c012cd]. For an isomer group with `Nc` conformers and `Na` atoms:

| Key | Shape | Dtype | Units | Universally populated? |
|---|---|---|---|---|
| `coordinates` | `(Nc, Na, 3)` | float32 | Å | yes |
| `atomic_numbers` | `(Na,)` | uint8 | — | yes |
| `wb97x_dz.energy` | `(Nc,)` | float64 | Ha | **yes (all ~5.5M conformers)** |
| `wb97x_dz.forces` | `(Nc, Na, 3)` | float32 | **Ha/Å** | **yes (all ~5.5M conformers)** |
| `ccsd(t)_cbs.energy` | `(Nc,)` | float64 | Ha | no (~500k subset = ANI-1ccx) |
| `wb97x_tz.forces` | `(Nc, Na, 3)` | float32 | Ha/Å | no (subset) |

[NB:74c012cd; Smith 2020 Table 1]

**Critical unit correction:** Forces are stored in **Hartree per Ångstrom (Ha/Å)**, NOT Hartree per Bohr [NB:74c012cd, Table 1, line "Atomic Forces 'wb97x_dz.forces' Ha/Å float32 (Nc, Na, 3)"]. Conversion to kcal/mol/Å multiplies by **627.5094740631 only**, with no Bohr-radius factor:

```
force_kcal_per_mol_per_angstrom = force_Ha_per_angstrom * 627.5094740631
```

A previous draft of this spec incorrectly included a `/0.5291772109` Bohr factor; that is **wrong** and would have scaled all reference forces by 0.529 (~half).

**The `iter_data_buckets` loader** (`aiqm/ANI1x_datasets/dataloader.py`) filters by requested keys: passing `data_keys=['wb97x_dz.energy', 'wb97x_dz.forces']` yields all ~5.5M conformers; adding `'ccsd(t)_cbs.energy'` filters to the ~500k ANI-1ccx subset [NB:74c012cd, "Usage Notes" Box 1].

**Implication for filter F4:** Since `wb97x_dz.forces` is universally populated, the previous-draft "F4 force-completeness ≥ 20 conformers" filter is effectively a no-op for ωB97X/6-31G* targets. Keep F4 as a defensive assertion (NaN-check), not as a gating filter.

---

## §4 Selection Criteria

### 4.1 Lane A — 16-molecule ANI-1x subset (primary topology heterogeneity)

**Filters (applied in sequence):**

| Step | Filter | Rationale | Source |
|---|---|---|---|
| F1 | Elements ⊆ {H=1, C=6, N=7, O=8} | ANI-1x guarantee | [NB:74c012cd] |
| F2 | **15 ≤ n_total_atoms ≤ 30** | Dipeptide-scale per §7.1; upper bound widened from 26→30 (ANI-1x max is 63, this captures AL5 tail) | [NB:65aa232f; AL5 amino acids/dipeptides] [NB:c0ac4c69; max=63] |
| F3 | `n_conf` ≥ 20 with forces non-NaN | Sufficient conformers for gradient averaging | — |
| F4 | `wb97x_dz.forces` non-NaN (defensive) | No-op for ωB97X/6-31G* but catches NaN edge cases | [NB:74c012cd] |
| F5 | SMILES parseable by RDKit | Canonical reproducibility | — |

### 4.2 Selection algorithm — AIMNet2 local-environment hashing

Random sampling on a 16-molecule subset risks selecting structurally redundant molecules. Adopt the **AIMNet2 local-environment hashing diversity strategy** (best precedent for small force-fitting subsets) [NB:6d670787; RSC d4sc08572h]:

> "For each atom, we utilized a hashing function operating on atomic number, number of connected hydrogen atoms, the total number of neighbors, and the same set of properties for all neighboring atoms. This hash uniquely encodes the local environment for each atom in a molecule, and comparing hash values was our strategy for discerning molecules with diverse chemical structures." [NB:6d670787]

**Algorithm `SELECT_16` (seed = 42):**

```
SELECT_16(hdf5_path, seed=42):
  1. Load passing_molecules via iter_data_buckets with F1-F5 applied
  2. For each molecule M:
       For each non-H atom a in M:
         hash(a) := (Z_a, n_H_connected, n_neighbors, sorted(Z_b for b in neighbors(a)))
       env_hash_set(M) := {hash(a) for a in M}
  3. global_hash_freq := Counter of all env_hash values across all passing molecules
  4. For each molecule M:
       rarity_score(M) := mean(1 / global_hash_freq[h] for h in env_hash_set(M))
  5. Sort passing_molecules by rarity_score descending (rarest envs first)
  6. Greedy diverse pick:
       selected := []
       picked_hashes := set()
       for M in sorted_molecules:
         if not env_hash_set(M).issubset(picked_hashes) and len(selected) < 16:
           selected.append(M)
           picked_hashes.update(env_hash_set(M))
  7. If len(selected) < 16: relax F3 to ≥10 conformers, restart
  8. Output selected sorted by n_total_atoms ascending then SMILES ascending
```

**Why this beats random:** With only 16 molecules, random sampling concentrates on the dense small-molecule region (mean 15 atoms); diversity selection forces representation of rare functional groups and rare bonded topologies, which is precisely what §7.1's "force-field fitting across varied topology" claim needs to demonstrate.

**Alternative considered and rejected:** Scaffold splitting (Murcko) is the standard precedent for *property prediction benchmarks* [NB:82d3eba5; DSpace MIT] but is designed for *train/test separation*, not for subset selection. AIMNet2-style diversity selection is the right tool for our 16-molecule force-fitting figure.

**Espaloma precedent (informational):** The closest prior force-field-fitting subset selection is espaloma + PepConf (capped/cyclic/disulfide peptides; valence-only fitting on gas-phase QM with zero-mean energy shift per molecule) [NB:bfb13e74]. Use espaloma's per-molecule zero-mean energy shift; do NOT use espaloma's random-by-molecule split (their use case is train/test, not subset selection).

### 4.3 Lane B — COMP6 cross-bucket supplement

Add **2-4 molecules from COMP6** spanning higher atom counts to demonstrate **cross-bucket** heterogeneous batching. All four COMP6 subsets use ωB97X/6-31G* with the same HDF5 schema as ANI-1x, so they drop into the pipeline with zero schema work.

| Subset | Atom count | Bucket index | Inclusion |
|---|---|---|---|
| COMP6 DrugBank | mean 44 ±20 atoms | bucket 0 (≤256) | optional 2 molecules — demonstrates within-bucket size range |
| COMP6 Tripeptides | mean 53 ±7 atoms | bucket 0 (≤256) | optional 2 molecules — peptide chemistry continuation of ANI-1x AL5 |
| COMP6 ANI-MD (small) | 20-100 atoms | bucket 0 | optional |
| **COMP6 ANI-MD (Trp-cage 1L2Y)** | **312 atoms** | **bucket 1 (1024)** | **mandatory — only molecule that crosses bucket boundary** |
| COMP6 ANI-MD (Chignolin 1UAO) | ~138 atoms | bucket 0 (≤256) | useful — 10-residue protein, still below the 256-atom bucket break |

**Recommended Lane-B set (4 molecules):** 1 DrugBank, 1 Tripeptide, 1 ANI-MD-drug, **1 Trp-cage**. Combined with the 16 Lane-A molecules → 20 total ensemble systems spanning 2 prolix buckets.

[NB:65aa232f; LANL Smith 2018 Table 1] [NB:65aa232f; "Trp-cage (1L2Y) and 10-residue Chignolin (1UAO)"]

### 4.4 Manifest format

`data/ani1x_subset/manifest.json`:

```json
{
  "spec_version": "2026-05-20-hp4-ani1x-curation.md v1.0",
  "lane_a": {
    "source": "ANI-1x",
    "source_doi": "10.6084/m9.figshare.10047041.v1",
    "source_sha256": "<filled at curation time>",
    "selection_algorithm": "AIMNet2-style local-environment hashing diversity (§4.2)",
    "selection_seed": 42,
    "force_unit_correction": "Ha/Å → kcal/mol/Å multiplier 627.5094740631 (no Bohr factor)",
    "molecules": [
      {
        "idx": 0, "smiles": "...", "n_total_atoms": 18,
        "n_heavy_atoms": 9, "n_conf_with_forces": 47,
        "file": "lane_a/mol_000.h5", "sha256": "...",
        "env_hashes": ["..."], "rarity_score": 0.0123
      }
    ]
  },
  "lane_b": {
    "source": "COMP6 (https://github.com/isayev/COMP6)",
    "source_sha256": "<filled at curation time>",
    "molecules": [
      {"idx": 0, "name": "Trp-cage_1L2Y", "subset": "ANI-MD",
       "n_total_atoms": 312, "n_conf": 128,
       "file": "lane_b/trp_cage.h5", "sha256": "..."}
    ]
  }
}
```

---

## §5 On-Disk Layout

```
data/
  ani1x_subset/
    manifest.json
    lane_a/
      mol_000.h5 ... mol_015.h5
    lane_b/
      drugbank_mol.h5
      tripeptide_mol.h5
      anim_drug.h5
      trp_cage.h5
```

Per-molecule HDF5 schema (same for Lane A and Lane B):

| Dataset | Shape | Dtype | Units |
|---|---|---|---|
| `positions` | `[N_conf, N_atoms, 3]` | float32 | Å |
| `forces` | `[N_conf, N_atoms, 3]` | float32 | **kcal/mol/Å** (unit-converted from raw Ha/Å) |
| `energy` | `[N_conf]` | float64 | Ha |
| `species` | `[N_atoms]` | int8 | atomic numbers |
| `smiles` | scalar | str | canonical SMILES (Lane A) / PDB ID (Lane B) |
| `molecule_id` | scalar | int64 | ANI-1x index OR COMP6 PDB-derived id |
| `lane` | scalar attr | str | "a" or "b" |
| `bucket_idx` | scalar attr | int | precomputed `_bucket_idx(n_total_atoms, ATOM_BUCKETS)` |

---

## §6 Differentiable Fitting Loss (for §7.1 implementation)

The curation spec specifies the loss formulation so the §7.1 implementer can write the gradient loop without re-research.

**Loss (per-molecule, then summed across the ensemble):**

```
L_m = (1/N_atoms_m) * mean[(F_pred - F_ref)^2]_atoms,conf
       + α * (1/N_conf_m) * mean[(E_pred - E_ref - shift_m)^2]_conf
       + w_reg * sum[(θ - θ_init)^2 / σ_θ^2]_params

L_total = sum_m L_m
```

**Parameters (research-backed):**

| Parameter | Value | Source |
|---|---|---|
| `α` (energy-to-force weight) | **0.25** (or 0.1 if force RMSE is paramount) | [NB:f0fc2a49; TorchANI dasher.wustl Table 3 — α=0.25 best forces 2.30/3.75] |
| `shift_m` (per-molecule energy offset) | mean(E_ref) − mean(E_pred), recomputed each step | [NB:bfb13e74; espaloma SI — "snapshot energies for each molecule shifted to zero mean"] |
| `w_reg` (parameter regularization weight) | **0.01** | [NB:3663adaa; ForceBalance / PMC9649520] |
| `σ_θ` (prior width per parameter type) | bond_k: 100 kcal/mol/Å²; bond_r0: 0.05 Å; angle_k: 30 kcal/mol/rad²; angle_θ0: 5° | [NB:3663adaa; harmonic-prior pattern] |
| Force loss normalization | per-atom (not per-molecule) | [NB:f0fc2a49] |
| Energy loss normalization | per-conformer, per-molecule | [NB:f0fc2a49] |

**Gradient stability:** If the §7.1 implementation later extends to `n_steps > 0` (differentiable simulation, NOT current scope), gradient truncation after 200 backward steps is the standard fix; gradients can explode even with stable forward MD [NB:7330dfce; arXiv 2412.04374]. The current spec uses `n_steps=0` so this is not load-bearing yet.

---

## §7 Fetch Script Spec

**Script:** `scripts/data/fetch_ani1x_subset.py` (no code in this spec; behavior only)

### 7.1 Responsibilities

1. Accept `--ani1x-archive PATH` and `--comp6-archive PATH`; require both. Verify SHA-256 against pinned constants. Fail fast on mismatch.
2. Run `SELECT_16` (§4.2) over ANI-1x with `--seed 42`.
3. Extract Lane B fixed molecules from COMP6: Trp-cage (mandatory) + 3 others per §4.3.
4. Convert forces Ha/Å → kcal/mol/Å using **multiplier 627.5094740631 only** (no Bohr factor — see §3 correction).
5. Write per-molecule HDF5 to `data/ani1x_subset/lane_a/` and `lane_b/`.
6. Compute SHA-256 per output file.
7. Emit `manifest.json` (schema in §4.4).
8. Print summary table.
9. Exit 0 on success.

### 7.2 Cluster constraint (CLUSTER.md §7)

Engaging compute nodes have no outbound internet. Workflow:

1. Local (internet machine): download ANI-1x from Figshare (§2) + COMP6 from GitHub. Verify SHA-256. Pin both in script.
2. Local Gate L1: `--dry-run` to verify imports + paths.
3. Local Gate L2: full run to produce `data/ani1x_subset/`.
4. `rsync -azP data/ani1x_subset/ engaging:~/projects/prolix/data/ani1x_subset/` — push curated subset only.
5. Cluster Gate L3: 1-molecule SLURM smoke job loads `mol_000.h5`, runs `EnsemblePlan(B=1).run(n_steps=0).gradient()`; verify non-NaN.

No runtime fetches. If `--ani1x-archive` path missing, print Figshare URL and exit non-zero.

### 7.3 Dependencies

`h5py`, `numpy`, `rdkit` (for canonical SMILES). No new deps. Vendor `iter_data_buckets` from `aiqm/ANI1x_datasets/dataloader.py` with attribution comment.

---

## §8 Exit Criteria

1. `sha256(ani1x_release.h5)` matches `ANI1X_EXPECTED_SHA256` in script.
2. `sha256(comp6_archive)` matches `COMP6_EXPECTED_SHA256` in script.
3. `data/ani1x_subset/lane_a/` has exactly 16 `mol_NNN.h5` files.
4. `data/ani1x_subset/lane_b/` has 4 files including `trp_cage.h5` (n_total_atoms=312, bucket_idx=1).
5. `manifest.json` reproduces hash-identical from a clean clone with seed 42 + both archives present.
6. All Lane A molecules have ≥ 20 conformers with non-NaN forces.
7. All species ∈ {1, 6, 7, 8}.
8. Force unit conversion verified: re-multiply forces by 1/627.5094740631 and check magnitudes are in Ha/Å range [0, ~0.5].
9. SHA-256 of each `mol_NNN.h5` agrees between local and cluster after rsync.
10. **Cluster Gate L3:** SLURM job loads `lane_b/trp_cage.h5`, constructs `MolecularBundle` via `make_bundle_from_system`, calls `EnsemblePlan(B=1).run(n_steps=0).gradient()` without NaN.
11. Span check: at least one Lane-B molecule has `bucket_idx > 0` (Trp-cage @ 312 atoms → `_bucket_idx(312, ATOM_BUCKETS)=1`). This is the cross-bucket evidence required by §7.1.

---

## §9 Risks

### R1 — No sulfur in ANI-1x; sulfur-containing dipeptides unavailable in primary

**Status:** Confirmed; mitigated by scope.

ANI-1x has only H, C, N, O [NB:74c012cd]. Sulfur-bearing residues (Cys, Met) cannot be fit from the primary dataset. **Mitigation:** §7.1 framing is "bonded fitting on heterogeneous CHNO organic ensemble"; sulfur is not load-bearing. If a referee demands S coverage, **SPICE Dipeptides (CC0, 33,850 conformers, full 676 dipeptide combinations including CYS-CYS disulfide)** is the natural extension [NB:73f75bd6] [NB:337e8307]. Cost: SPICE uses ωB97M-D3(BJ)/def2-TZVPPD — a different reference baseline; would require either re-running ANI-1x at SPICE level or scoping a SPICE-only follow-up figure. Not the right move pre-submission.

### R2 — Cross-bucket heterogeneity demonstration

**Status:** **Resolved** by COMP6 supplement.

Previously flagged as "Certain" risk: all ANI-1x molecules fit in `ATOM_BUCKETS[0]=256`. **Resolution:** COMP6 ANI-MD Trp-cage (1L2Y, 312 atoms, `wb97x_dz.forces` present at ωB97X/6-31G* — same DFT level as ANI-1x) crosses into `ATOM_BUCKETS[1]=1024` [NB:65aa232f]. The 20-molecule (16 Lane A + 4 Lane B) ensemble spans 2 buckets. §7.1 can now claim *both* topology heterogeneity (within bucket 0) and *atom-count-bucket* heterogeneity (across 0→1).

### R3 — Force completeness in upper tail of ANI-1x

**Status:** Confirmed non-issue.

Previous draft worried that AL5 amino-acid/dipeptide additions might have sparse `wb97x_dz.forces`. **Research finding:** `wb97x_dz.forces` is **universally populated** across all ~5.5M ANI-1x conformers; only the CCSD(T) and triple-zeta keys are sparse [NB:74c012cd]. F3 (≥20 conformers) and F4 (force non-NaN) remain in the spec defensively, but both are essentially no-ops for ωB97X/6-31G* targets.

### R4 — Force unit conversion error

**Status:** Caught during research; corrected.

Previous draft used `force × 627.5094740631 × (1/0.5291772109)` (treating storage as Ha/Bohr). **Correct conversion:** ANI-1x stores `wb97x_dz.forces` in **Ha/Å** directly [NB:74c012cd, Table 1], so the multiplier is `627.5094740631` *only*. The §8 exit criterion 8 verifies this by round-tripping.

### R5 — Diversity selection may collapse on functional-group near-duplicates

**Status:** Open; mitigated by exit-criterion review.

AIMNet2's hash function encodes Z + n_H + n_neighbors + sorted neighbor Z's. Two molecules that differ only in skeleton length but have similar local environments will get overlapping hashes. **Mitigation:** Exit criterion review must visually inspect the 16 selected SMILES and confirm chemical diversity (different scaffolds, different functional groups present). If insufficient, fall back to scaffold-stratified diversity with seed=42.

---

## §10 Integration with Prolix

### 10.1 Bundle construction (per molecule)

```
for mol_path in glob('data/ani1x_subset/lane_*/*.h5'):
    with h5py.File(mol_path) as f:
        positions = jnp.array(f['positions'][0])     # [N_atoms, 3]
        species = jnp.array(f['species'])             # [N_atoms]
        forces_ref = jnp.array(f['forces'][0])        # [N_atoms, 3]
        bucket_idx = f.attrs['bucket_idx']
        smiles = f['smiles'][()].decode()
    # Bonded topology from SMILES via RDKit + GAFF2/MMFF94 initial params
    system = build_system_from_smiles(smiles, positions, species)
    bundle = make_bundle_from_system(system, boundary_condition='free')
```

### 10.2 §7.1 gradient loop (pseudocode)

```
bundles = [load_bundle(p) for p in mol_paths]          # 20 bundles
forces_ref = [load_forces(p) for p in mol_paths]       # 20 force arrays
plan = EnsemblePlan.from_bundles(bundles)              # Claim 1 hetero-batch

def loss_fn(ff_params):
    predicted = plan.run(n_steps=0, ff_params=ff_params).gradient()
    # per §6 loss spec
    return sum(per_mol_loss(p, r, ff_params) for p, r in zip(predicted, forces_ref))

grad_fn = jax.grad(loss_fn)
```

### 10.3 Conformer axis

Lane A molecules have many conformers per file (e.g., 47); Lane B Trp-cage has 128. Use one conformer per molecule per gradient step for the headline figure; report wall-clock as median across 3 seeds for conformer selection.

---

## §11 References

**Primary papers (notebook sources):**
- Smith et al. 2020 — ANI-1ccx/ANI-1x [NB:74c012cd; PMC7195467]
- Smith et al. 2018 — ANI-1x development + COMP6 [NB:65aa232f; LANL "Less is more"]
- Eastman et al. 2023 — SPICE dataset [NB:73f75bd6] [NB:337e8307]
- Anstine et al. 2025 — AIMNet2 + local-env hashing [NB:6d670787; RSC d4sc08572h]
- Gao et al. 2020 — TorchANI + loss formulation [NB:f0fc2a49; pubs.acs.org/jcim]
- Wang et al. 2022 — espaloma [NB:bfb13e74; semantic-scholar eabf]
- Pugliese et al. 2022 — ForceBalance reproducibility [NB:3663adaa; PMC9649520]
- Schoenholz & Cubuk 2020 — JAX-MD [NB:3449e5b3; NeurIPS]
- Greener 2024 — Reversible DMS [NB:7330dfce; arXiv 2412.04374]

**Notebook (full source list):** `301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76` — 39 curated sources.

*End of spec.*
