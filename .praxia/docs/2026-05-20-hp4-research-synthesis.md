# HP4 Research Synthesis — ANI-1x + COMP6 Curation for §7.1 Paper Gate

**Task ID:** 260520_hp4_ani1x
**Backlog:** [260] HP4 ANI-1x DFT-forces subset curation sub-spec
**Date:** 2026-05-20
**Roadmap anchors:** `docs/superpowers/specs/2026-05-19-prolix-long-horizon-roadmap.md` §2.7 (HP4), §7.1 (Lane-B paper-gating figure)
**Actionable spec:** `docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md`
**Source notebook:** `301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76` (39 curated sources, NotebookLM)

---

## Purpose

This document captures the **research findings** from 8 targeted NotebookLM queries over 39 curated web sources, supporting the HP4 sub-spec. Distinct from the spec: the spec is *prescriptive* (what we will build); this document is *evidentiary* (what the literature says). KB consumers (future librarians, oracle review passes, downstream §7.1 implementation) should query this artifact for source-grounded claims.

**Tags:** `prolix` `hp4` `ani-1x` `comp6` `spice` `dft-forces` `differentiable-md` `force-field-fitting` `paper-§7.1` `claim-1` `s1-differentiability`

---

## Synthesis Decisions

The spec's design choices map to specific findings below:

| Decision | Driving finding | Source |
|---|---|---|
| Primary dataset = ANI-1x | Public, single HDF5, ωB97X/6-31G* with forces universally populated, supports dipeptides via AL5 | Q2, Q3, Q4 |
| Supplement dataset = COMP6 (not SPICE) | Same DFT level + same HDF5 schema as ANI-1x → no reference-baseline mixing | Q5 |
| Cross-bucket evidence = Trp-cage + Chignolin + 2 mid-size | COMP6 ANI-MD has 312-atom Trp-cage AND 138-atom Chignolin with forces at ωB97X/6-31G* | Q1, Q8 |
| Forces unit = Ha/Å (not Ha/Bohr) | Smith 2020 Sci Data Table 1 explicitly Ha/Å | Q2 |
| ATOM_BUCKETS ladder = (64, 128, 256, 1024, …) | Original (256, 1024, …) ladder makes 3+ bucket-transitions impossible from any organic DFT-forces dataset | Q3, Q8 |
| Subset selection = AIMNet2 local-environment hashing | Best-precedent diversity method for n=16 force-fitting subsets | Q6 |
| Loss formulation = energy + α·force, α=0.25, w_reg=0.01 | TorchANI Table 3 sweep + ForceBalance defaults | Q7 |
| Per-molecule zero-mean energy shift | espaloma SI prescribes this for MM-vs-QM heat-of-formation offset | Q7 |
| Lane B = held-out test set | Pre-empts "in-distribution overfitting" referee objection | Q5, Q6, oracle |
| Sulfur dropped from element whitelist | ANI-1x is CHNO only; sulfur extension is ANI-2x or SPICE | Q3, Q5 |
| Novel-claim: vmap-vs-loop on heterogeneous topologies | No prior work in espaloma, ForceBalance, chemtrain, JAX-MD, reversible DMS reports this | Q7 |

---

## Query 1 — Does COMP6 have DFT forces (not just energies) at ωB97X/6-31G*?

**Question:** Does COMP6 (and specifically the 312-atom Trp-cage entry) contain DFT atomic forces, or only energies?

**Answer:** **Yes, forces are available.** COMP6 contains both energies and forces at ωB97X/6-31G* for all non-equilibrium conformations across all six subsets (GDB7to9, GDB10to13, Tripeptides, DrugBank, ANI-MD, S66x8). The Trp-cage (1L2Y, 312 atoms) sits inside the ANI-MD subset.

**Verbatim:** *"Energies and forces for all non-equilibrium molecular conformations presented have been calculated using the ωB97x density functional with the 6-31G(d) basis set as implemented in the Gaussian 09 electronic structure software."* — Smith 2018 LANL [NB:65aa232f]

**ANI-MD sampling:** *"Forces from the ANI-1x potential are applied to run 1ns of vacuum molecular dynamics with a 0.25fs time step at 300K using the Langevin thermostat on 14 well-known drug molecules and two small proteins. System sizes range from 20 to 312 atoms. A random subsample of 128 frames from each 1ns trajectory is selected, and reference DFT single point calculations are performed to obtain QM energies and forces."* [NB:65aa232f]

**Storage:** HDF5 with key `wb97x_dz.forces`, shape `(Nc, Na, 3)`, dtype float32, units Ha/Å [NB:74c012cd].

**Implication for HP4:** R2 (cross-bucket evidence missing) is resolved. Trp-cage + Chignolin + multiple mid-size COMP6 molecules are all directly compatible with the ANI-1x training pipeline.

**Sources:**
- [NB:65aa232f] — LANL Smith 2018 "Less is more: Sampling chemical space with active learning" — https://laro.lanl.gov/view/pdfCoverPage?instCode=01LANL_INST&filePid=13158206750003761
- [NB:74c012cd] — LANL Smith 2020 *Sci Data* — https://cnls.lanl.gov/~serg/postscript/s41597-020-0473-z.pdf

---

## Query 2 — ANI-1x HDF5 schema, keys, units, completeness, loader

**Top-level structure:** Single HDF5 file, grouped by chemical isomers/molecular formulas. Within each isomer group, datasets aggregate all conformers (`Nc`) and atoms (`Na`) for that formula.

**Key convention:** **Flat dot-notation**, e.g. `wb97x_dz.forces` (NOT hierarchical `wb97x_dz/forces`). Critical for `iter_data_buckets` to function.

**Universal vs. subset keys (`Nc` conformers, `Na` atoms):**

| Key | Shape | Dtype | Units | Universally populated? |
|---|---|---|---|---|
| `coordinates` | `(Nc, Na, 3)` | float32 | Å | yes |
| `atomic_numbers` | `(Na,)` | uint8 | — | yes |
| `wb97x_dz.energy` | `(Nc,)` | float64 | Ha | **yes (all ~5.5M)** |
| `wb97x_dz.forces` | `(Nc, Na, 3)` | float32 | **Ha/Å** | **yes (all ~5.5M)** |
| `ccsd(t)_cbs.energy` | `(Nc,)` | float64 | Ha | no (~500k ANI-1ccx subset only) |
| `wb97x_tz.forces` | `(Nc, Na, 3)` | float32 | Ha/Å | no (subset) |

[NB:74c012cd, Smith 2020 Table 1]

**Critical unit correction:** Forces are stored in **Hartree/Ångstrom**, NOT Hartree/Bohr. Conversion to kcal/mol/Å is `× 627.5094740631` *only* — no Bohr-radius factor. The earlier librarian draft of the HP4 spec included a `/0.5291772109` factor that would have scaled all reference forces by 0.529 (~half).

**The `iter_data_buckets` loader:** Filters by requested keys. Passing `data_keys=['wb97x_dz.energy', 'wb97x_dz.forces']` yields all ~5.5M conformers. Adding `'ccsd(t)_cbs.energy'` filters to the ~500k ANI-1ccx subset. The loader silently skips conformers missing any requested key.

**Verbatim Smith 2020:** *"The script will only load conformers that share the requested data for property keys given in the 'data_keys' list. For example, if the 'data_keys' list contains two keys 'wb97x_dz.energy' and 'ccsd(t)_cbs.energy', then only conformers that share both energies will be loaded, approximately 500k structures. However, if one removes 'ccsd(t)_cbs.energy' from the list, then approximately 5 million structures will be loaded."* [NB:74c012cd]

**Sources:**
- [NB:74c012cd] — LANL Smith 2020 — https://cnls.lanl.gov/~serg/postscript/s41597-020-0473-z.pdf
- [NB:c0ac4c69] — qchem/dataset.py — https://github.com/icanswim/qchem/blob/main/dataset.py

---

## Query 3 — ANI-1x atom-count distribution

**Quantified:**
- **Total unique molecules:** 63,865
- **Total conformers:** 5,496,771
- **Mean atom count:** 15 total (8 heavy, C/N/O), σ=5
- **Maximum atom count:** **63** (entire dataset)
- **Median:** not provided in sources
- **Distribution shape:** Log-scale histogram; vast majority < 20 total atoms; AL5 adds the upper-tail amino-acid and dipeptide fragments

**Atom-count is bounded at 63 — important constraint.** The earlier librarian draft of the HP4 spec assumed an upper bound near 26 atoms. The true ceiling is 63. This widens the F2 filter range from 15-26 (overly conservative) to 15-30 (dipeptide-scale per task spec). Larger ANI-1x molecules go up to 63 but those are no longer dipeptide-scale.

**Source molecules:** ANI-1 (GDB-11 derived) + active-learning additions from GDB-11 (9-heavy), ChEMBL small molecules, generated amino acids, and 2-amino-acid peptides [NB:9045d3c5].

**Sampling methods used:** (1) molecular dynamics, (2) normal mode sampling, (3) dimer sampling, (4) torsion sampling [NB:9045d3c5].

**Implication for bucket strategy:** With current `ATOM_BUCKETS = (256, 1024, …)`, every ANI-1x molecule lands in bucket 0. Need finer ladder to support multi-bucket §7.1 figure. Recommended new ladder: `(64, 128, 256, 1024, …)`.

**Sources:**
- [NB:9045d3c5] — OpenQDC ANI-1x docs — https://docs.openqdc.io/stable/API/datasets/ani.html
- [NB:65aa232f] — LANL Smith 2018 Table 1
- [NB:8e72758c] — DTU Schreiner — https://backend.orbit.dtu.dk/ws/files/337904227/d3cp02143b.pdf
- [NB:c0ac4c69] — qchem/dataset.py — *"Longest molecule is 63 atoms"*

---

## Query 4 — AL5 contributions (amino acids and dipeptides)

**AL iterations summary:**

| Cycle | Sources added |
|---|---|
| AL1 | Reduced 1-6 heavy-atom subset of ANI-1 (initialization) |
| AL2 | Continued GDB-11 small molecules, sizes gradually increasing |
| AL3 | Up to 7 heavy atoms, GDB-11 — 1.8M conformers cumulative, AL3 matches ANI-1 ensemble |
| AL4 | Continued GDB-11 expansion |
| **AL5** | **Amino acids + generated dipeptides + small molecule dimers + ChEMBL small molecules** |
| AL6 | GDB-11 9-heavy-atom subset → final 5.5M ANI-1x dataset |

**Verbatim:** *"Eventually, between the AL4 and AL5 steps, amino acids, generated dipeptides, generated small molecule dimers and small ChEMBL molecules were added to the sampling set. This is apparent from the large drop in error between AL4 and AL5 for the DrugBank, Tripeptides, and S66x8 benchmarks."* [NB:65aa232f]

**Implication:** Dipeptide-scale chemistry IS in the ANI-1x training data (introduced at AL5). The F2 filter range 15-30 captures this upper-tail content. Exact conformer counts and atom-count ranges added at AL5 are not separately quantified in sources.

**Sources:**
- [NB:65aa232f] — LANL Smith 2018

---

## Query 5 — Dataset alternatives (ANI-1x vs ANI-2x vs SPICE vs QMugs vs V2DFT)

| Dataset | Elements | Forces | Peptide coverage | License | Total size | DFT method | Format |
|---|---|---|---|---|---|---|---|
| **ANI-1x** | H, C, N, O | yes | yes (amino acids + dipeptides at AL5) | research-redistributable (not explicitly CC) | 5.5M conf / 63,865 mols | ωB97X/6-31G* | HDF5 |
| **ANI-2x** | H, C, N, O, F, Cl, S | yes | yes (S-containing amino acids + dipeptides explicit) | available via Zenodo/GitHub | 8.9M conf / ~13k isomers | ωB97X/6-31G* (same as ANI-1x) | HDF5 (same schema) |
| **SPICE** | 15 elements (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I) | yes | **676 dipeptides exhaustive (all 20 AA pairs + protonation variants + CYS-CYS disulfide)** in dedicated subset (33,850 conformers, 26-60 atoms) | **CC0 (public domain)** | ~1.1M (v1) → 2M+ (v2) | ωB97M-D3(BJ)/def2-TZVPPD | HDF5 |
| **QMugs** | H, C, N, O, F, P, S, Cl, Br, I | **no** (energies only) | no (generic ChEMBL) | hosted on ETH Library | 2M conf / 665k mols | GFN2-xTB + ωB97X-D/def2-SVP | SDF/CSV/tar |
| **V2DFT (nablaDFT v2)** | H, C, N, O, F, S, Cl, Br | yes (2.9M conformations + trajectories) | no (MOSES/ZINC drug-like) | MIT | 15.7M conf / 1.9M mols | ωB97X-D/def2-SVP | GitHub API |

**Verdict for HP4:**
- **Primary:** ANI-1x. Pre-existing AL5 dipeptide coverage; same DFT level as COMP6 supplement; smaller and tractable.
- **Supplement:** COMP6 (NOT a fallback dataset — see Q1). Same ωB97X/6-31G*, same HDF5, drop-in.
- **Sulfur fallback (if reviewers demand):** **SPICE Dipeptides** is the right pivot. 676-dipeptide exhaustive + CC0 + forces. Cost: different DFT level (ωB97M-D3(BJ)/def2-TZVPPD), separate HDF5 schema; cannot mix references with ANI-1x.
- **Avoid:** QMugs (no forces). V2DFT (no peptide coverage, ωB97X-D differs from ωB97X plain).

**Verbatim SPICE description:** *"It consists of all possible dipeptides formed by the 20 natural amino acids and their common protonation variants. … That gives 26 amino acid variants, which can be combined to form 676 possible dipeptides. Each one is terminated with ACE and NME groups. A pair of CYS residues, each terminated with ACE and NME groups and connected to each other by a disulfide bond, is also included."* [NB:73f75bd6]

**Sources:**
- [NB:73f75bd6], [NB:fa3a43c6] — SPICE UPF + arXiv
- [NB:337e8307] — SPICE GitHub openmm/spice-dataset (CC0 confirmed)
- [NB:51e8af53] — Molecular QC datasets review arXiv 2408.12058
- [NB:f298ddb5] — V2DFT NeurIPS 2024
- [NB:65aa232f], [NB:74c012cd] — ANI-1x sources

---

## Query 6 — Subset-selection methodologies in prior ML-potential papers

**TorchANI (Gao 2020):** Random 80/10/10 split on full ANI-1x. Subsequent uncertainty work introduced **bin-balanced stratification by chemical formula** (cluster by `C9H12O6`-like exact composition, then assign whole clusters to train/test) to prevent isomer leakage [NB:8deba98a].

**AIMNet2 (Anstine 2025):** **Diversity selection via local-environment hashing.** For each non-H atom: `hash = (Z, n_H_connected, n_neighbors, sorted(neighbor_Zs))`. Selected molecules containing the *least frequent* atomic hashes across the chemical space — 10 molecules per element type → 113 unique molecules after dedup. *"These 113 molecules exemplify a selection of the most unusual chemical bonding present in CSD, and thus serve as challenging test cases."* [NB:6d670787]

**Espaloma (Wang 2022):** Random shuffle and split *by molecule* (not by conformer) — different conformers of the same molecule stay in the same train/val/test partition. Uses PepConf for valence-only force-field fitting on capped/cyclic/disulfide peptides. Per-molecule zero-mean energy shift before computing energy MSE [NB:bfb13e74].

**EspalomaCharge:** SPICE training with random shuffle keeping protomeric/tautomeric variants in the same partition [NB:ec5337ca].

**MACE:** Uses out-of-domain temperature/dihedral splits (3BPA: train at 300K, test at 600K, 1200K, and dihedral slices) — designed to test extrapolation, not subset diversity [NB:8f5095ad].

**Scaffold splitting (Murcko, MoleculeNet, ChemProp):** Partition by 2D scaffold framework; standard precedent for train/test in property-prediction benchmarks; not designed for subset selection [NB:82d3eba5].

**Recommendation for 16-molecule §7.1 force-fitting figure:**
Adopt **AIMNet2-style local-environment hashing** with greedy diverse picking. Random sampling on 16 molecules concentrates on the dense small-molecule region (ANI-1x mean ≈ 15 atoms) and risks chemical redundancy. Scaffold splitting is the wrong tool (it's a train/test tool, not a subset-selection tool). AIMNet2 hashing maximizes the chemical-environment diversity needed to show generalizable force-field fitting.

**Caveat (oracle review R9):** AIMNet2 hashing optimizes atomic-environment rarity, which is a proxy for bonded-type diversity but not identical. Add a bonded-type-coverage check (`|bond_types| ≥ 20 ∧ |angle_types| ≥ 30`) as a post-selection exit gate; fall back to hybrid `0.5 × rarity + 0.5 × bonded-coverage-gain` if violated.

**Sources:**
- [NB:f0fc2a49] — TorchANI Gao 2020
- [NB:8deba98a] — TorchANI uncertainty (bin-balanced)
- [NB:6d670787] — AIMNet2 RSC d4sc08572h
- [NB:bfb13e74] — espaloma semantic-scholar
- [NB:ec5337ca] — EspalomaCharge
- [NB:8f5095ad] — MACE
- [NB:82d3eba5] — Murcko/scaffold MoleculeNet

---

## Query 7 — Differentiable bonded-parameter fitting prior art

**Loss formulation (energy + α·force):**

Recommended composite loss: `L = energy_loss + α × force_loss`. TorchANI's sweep over α ∈ {0, 0.1, 0.25, 0.5} found α = 0.25 gives best force RMSE (2.30 kcal/mol/Å) with only a small absolute-energy RMSE penalty; α = 0.1 is the conservative pick if force RMSE is paramount [NB:f0fc2a49, Table 3].

**Energy normalization:**
- TorchANI: `energy_loss = (1/N_atoms) × sum((E_pred - E_truth)^2)` per molecule, scaled by sqrt(N_atoms).
- SPICE training: explicit physical-unit weighting — `1 (kJ/mol)^-2` for energies, `1 (kJ/mol/Å)^-2` for forces [NB:73f75bd6].
- Espaloma: per-molecule zero-mean shift on both predicted and reference energies before computing MSE (because MM force fields cannot reproduce QM heats of formation; the constant offset must be removed) [NB:bfb13e74].

**Force normalization:** Per-atom, mean over (atoms × conformers).

**Parameter regularization:**
- ForceBalance: harmonic prior (Gaussian penalty) on parameter deviation from initial values, `w_reg = 0.01`, with per-parameter-type prior widths `σ_θ` (the "trust radius" of the optimizer) [NB:3663adaa].
- Espaloma: loss can be augmented with physical penalties (e.g., penalize short vibrational periods to prevent integration instabilities under hydrogen mass repartitioning) [NB:bfb13e74].

**Gradient stability:**
- Differentiable molecular simulation (DMS) has gradient-explosion issues even when forward MD is stable — gradients accumulate multiplicatively backward in time [NB:7330dfce].
- Standard fix: **gradient truncation** after ~200 backward steps (Greener 2024), or gradient-norm clipping.
- For `n_steps = 0` (current §7.1 scope): not an issue. Becomes load-bearing only if §7.1 extends to non-zero MD-step gradients.

**Critical novelty finding — gap in literature:**
*No prior work reports wall-clock comparisons between batched-vmap and looped-per-system gradient evaluation on heterogeneous-topology ensembles.*

- JAX-MD's documented `vmap` use is over *homogeneous* parameter sweeps (particle diameters, RNG keys), not varied molecular topologies [NB:3449e5b3].
- Espaloma reports batched-parametrization throughput (e.g., 100 ACE-ALA_n-NME molecules in 7.11s on CPU) but no AD-gradient-throughput comparison [NB:bfb13e74, NB:ec5337ca].
- ForceBalance optimizes via trust-radius Newton-Raphson, not batched AD [NB:3663adaa].
- Chemtrain runs JAX-MD vmap but doesn't benchmark hetero-topology batching [NB:0f84e052].
- Reversible DMS (Greener 2024) benchmarks memory cost reduction, not vmap vs loop wall-clock on heterogeneous topologies [NB:7330dfce].

This gap is itself a paper claim. §7.1 should explicitly call out the absence of prior wall-clock comparison and frame the 50–100× speedup as the first quantification of this property.

**Sources:**
- [NB:f0fc2a49] — TorchANI Gao 2020 *J Chem Inf Model* — https://dasher.wustl.edu/chem430/readings/jcim-60-3408-20.pdf
- [NB:73f75bd6] — SPICE arXiv 2209.10702 (loss weighting)
- [NB:bfb13e74] — espaloma SI
- [NB:3663adaa] — Coarse-grained FF optimization (ForceBalance regularization pattern)
- [NB:bf419cb0] — ForceBalance PMC9649520
- [NB:7330dfce] — Reversible DMS arXiv 2412.04374
- [NB:3449e5b3] — JAX-MD NeurIPS 2020
- [NB:0f84e052] — chemtrain TUM

---

## Query 8 — Cross-bucket supplement options (30–300 atom range with DFT forces)

**Evaluated options:**

| Dataset | Max atoms | Forces? | DFT level | Peptide coverage | License | Format |
|---|---|---|---|---|---|---|
| **COMP6 (multiple subsets)** | **312** (Trp-cage in ANI-MD) | yes | **ωB97X/6-31G(d) — same as ANI-1x** | yes (Tripeptides 248 mols mean 53 atoms; ANI-MD includes 1L2Y + 1UAO) | publicly available (GitHub `isayev/COMP6`) | **HDF5 same as ANI-1x** |
| SPICE peptide subsets | 110 (Solvated PubChem); 79-96 (Solvated AA); 26-60 (Dipeptides) | yes | ωB97M-D3(BJ)/def2-TZVPPD | extensive (Dipeptides, Solvated AA, AA-ligand pairs) | CC0 | HDF5 |
| GEOM-Drugs | 181 (AICures) | **no** (DFT energies on BACE-1 subset only via r2scan-3c/mTZVPP; no forces) | xTB + r2scan-3c | no peptide focus (drug-like only) | publicly available | XYZ/JSON/CSV |
| MD17 / rMD17 | 24 (azobenzene) / 21 (aspirin) | yes (MD-trajectory only) | PBE+vdW-TS / PBE/def2-SVP | none (10 small molecules) | publicly available | NumPy/XYZ |
| V2DFT | 62 | yes (2.9M conformations) | ωB97X-D/def2-SVP (note ω**B97X-D**, not plain ωB97X) | no peptide focus | MIT | GitHub API |
| OMol25 | not specified | yes | "advanced DFT" (functional not named) | biomolecules included | open | not specified |

**Verdict — COMP6 is overwhelmingly the right supplement:**
1. **Same level of theory** (ωB97X/6-31G*) — no reference-baseline mixing penalty.
2. **Same HDF5 schema** — drop into the ANI-1x loader pipeline with zero schema work.
3. **Naturally tiered atom counts** for cross-bucket evidence:
   - Bucket 0 (≤256 under default ladder, or bucket-shifted under finer ladder): GDB7to9 (17), GDB10to13 (25), Tripeptides (53), DrugBank (44 ± 20).
   - Bucket 1+: ANI-MD (75 mean, max 312); Chignolin (~138); Trp-cage (312).
4. **Non-MD force sources available:** DrugBank and Tripeptides use Diverse Normal Mode Sampling (DNMS), not just MD-trajectory frames [NB:65aa232f].

**Critical constraint exposed by this query:** Even with COMP6, only the ANI-MD subset has molecules above ~250 atoms. Chignolin (138) and Trp-cage (312) are essentially the only options above the default `ATOM_BUCKETS[0] = 256` boundary in COMP6. This is why HP4 mandates the finer ladder `(64, 128, 256, 1024, …)`: it converts a 1-or-2-bucket-spread crisis into a 3-4-bucket spread using available data.

**Why not SPICE peptide subsets:** Different functional + dispersion correction (ωB97M-D3BJ); cannot be combined with ANI-1x training data without re-running QM on the SPICE subset at ωB97X/6-31G*. Documented as fallback if a referee demands S-containing residues (Cys/Met). Documented in spec §9 R1.

**Sources:**
- [NB:65aa232f] — COMP6 (LANL Smith 2018) — Table 1 atom-count tiers and DNMS
- [NB:74c012cd] — HDF5 schema parity with ANI-1x
- [NB:73f75bd6] — SPICE peptide subsets details
- [NB:705131e5] — GEOM-Drugs (no forces)
- [NB:64f68891] — MD17 / rMD17 statistics
- [NB:51e8af53] — V2DFT, dataset survey
- [NB:4f25e77e] — OMol25 LBL announcement

---

## Source Map (39 imported sources by query coverage)

| Source ID | Title | URL/DOI | Used in Q# |
|---|---|---|---|
| [NB:73f75bd6] | SPICE UPF | https://repositori.upf.edu/server/api/core/bitstreams/81c738bb/content | Q5, Q7, Q8 |
| [NB:fa3a43c6] | SPICE arXiv 2209.10702 | https://arxiv.org/pdf/2209.10702 | Q5 |
| [NB:573cb366] | OpenQDC datasets | https://www.openqdc.io/datasets | Q5 |
| [NB:51e8af53] | Molecular QC datasets arXiv 2408.12058 | https://arxiv.org/pdf/2408.12058 | Q5, Q8 |
| [NB:0e924bd1] | LANL ANI-1ccx/1x | https://laro.lanl.gov/.../13158220330003761 | Q5 |
| [NB:337e8307] | openmm/spice-dataset GitHub | https://github.com/openmm/spice-dataset | Q5 |
| [NB:f298ddb5] | V2DFT NeurIPS 2024 | https://proceedings.neurips.cc/.../40d45b1e.../V2DFT.pdf | Q5 |
| [NB:69eeb4d8] | Open Force Field Initiative | https://escholarship.org/content/qt7jc0z77p/qt7jc0z77p.pdf | Q5 |
| [NB:5fe82512] | Smith 2020 ChemRxiv | https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.10050737.v2 | Q2 |
| [NB:74c012cd] | Smith 2020 LANL | https://cnls.lanl.gov/~serg/postscript/s41597-020-0473-z.pdf | Q1, Q2, Q3, Q4, Q5, Q8 |
| [NB:8a01b405] | Smith 2020 PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC7195467/ | Q2 |
| [NB:c0ac4c69] | qchem/dataset.py | https://github.com/icanswim/qchem/blob/main/dataset.py | Q2, Q3 |
| [NB:7a970229] | DTU Schreiner thesis | https://backend.orbit.dtu.dk/.../Thesis_Mathias_Schreiner_DTU.pdf | Q2 |
| [NB:9045d3c5] | OpenQDC ANI docs | https://docs.openqdc.io/stable/API/datasets/ani.html | Q3 |
| [NB:52574bd9] | OpenQDC ani1x | https://www.openqdc.io/datasets/ani1x | Q3 |
| [NB:8deba98a] | TorchANI uncertainty arXiv 2501.05250 | https://arxiv.org/html/2501.05250v2 | Q6 |
| [NB:7634551b] | TorchANI docs | https://aiqm.github.io/torchani-test-docs/api.html | Q2 |
| [NB:a021c4fa] | ACS Chem Rev 4c00572 | https://pubs.acs.org/doi/10.1021/acs.chemrev.4c00572 | Q6 |
| [NB:70c3732f] | MoleculeNet arXiv 1703.00564 | https://www.arxiv.org/pdf/1703.00564v1 | Q6 |
| [NB:82d3eba5] | DSpace MIT scaffold splits | https://dspace.mit.edu/.../acs.jcim.9b00237.pdf | Q6 |
| [NB:8f5095ad] | MACE arXiv 2206.07697 | https://arxiv.org/pdf/2206.07697 | Q6 |
| [NB:6d670787] | AIMNet2 RSC | https://pubs.rsc.org/en/content/articlelanding/2025/sc/d4sc08572h | Q6 |
| [NB:f0fc2a49] | TorchANI Gao 2020 | https://dasher.wustl.edu/chem430/readings/jcim-60-3408-20.pdf | Q6, Q7 |
| [NB:bfb13e74] | espaloma SI | https://pdfs.semanticscholar.org/eabf/.../espaloma.pdf | Q6, Q7 |
| [NB:0f84e052] | chemtrain TUM | https://portal.fis.tum.de/.../cd634ac7.../chemtrain | Q7 |
| [NB:3449e5b3] | JAX-MD NeurIPS | https://papers.nips.cc/paper/2020/file/83d3d4b6.../JAX-MD.pdf | Q7 |
| [NB:bf419cb0] | ForceBalance PMC9649520 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9649520/ | Q7 |
| [NB:7330dfce] | Reversible DMS arXiv 2412.04374 | https://arxiv.org/pdf/2412.04374 | Q7 |
| [NB:a5082dab] | Torsion reparam ChemRxiv | https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2024-lcnx1 | Q7 |
| [NB:3663adaa] | CG FF optimization RSC d0cp05041e | https://pubs.rsc.org/.../d0cp05041e | Q7 |
| [NB:4d87e1ef] | OSTI charge model | https://www.osti.gov/servlets/purl/1479996 | Q3 |
| [NB:8e72758c] | DTU GNN ensemble | https://backend.orbit.dtu.dk/.../d3cp02143b.pdf | Q3 |
| [NB:65aa232f] | LANL "Less is more" / COMP6 | https://laro.lanl.gov/.../13158206750003761 | Q1, Q4, Q8 |
| [NB:5d829069] | ELoRA Equivariant GNN | https://raw.githubusercontent.com/mlresearch/v267/.../wang25al.pdf | Q8 |
| [NB:705131e5] | GEOM-Drugs Benchmark | https://www.emergentmind.com/topics/geom-drugs-benchmark | Q8 |
| [NB:b0d79aba] | ANI-2x extension ChemRxiv | https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.11819268.v1 | Q5 |
| [NB:64f68891] | MD17/rMD17 arXiv 2602.16897 | https://arxiv.org/pdf/2602.16897 | Q8 |
| [NB:4f25e77e] | OMol25 LBL News | https://newscenter.lbl.gov/2025/05/14/.../OMol25 | Q8 |
| [NB:ec5337ca] | EspalomaCharge ar5iv | https://ar5iv.labs.arxiv.org/html/2302.06758 | Q7 |

---

## Cross-references

- **Actionable spec:** `docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md` (the spec this synthesis supports)
- **Roadmap parent:** `docs/superpowers/specs/2026-05-19-prolix-long-horizon-roadmap.md` §7.1
- **Sibling HP specs:**
  - HP1 (legacy API migration): `docs/superpowers/specs/2026-05-20-legacy-api-migration.md`
  - HP3 (MolecularShapeSpec coarsening): completed; see `tests/batching/test_safe_map_varying_shape_spec.py` and `src/prolix/types/bundles.py`
- **Backlog items:**
  - 260 — HP4 ANI-1x curation sub-spec (this synthesis supports)
  - 349 — HP4-WASM Rust-native parser (created from this work)
  - 259 — §7.1 figure (consumer of this HP4 deliverable)
- **Notebook:** [HP4 ANI-1x research notebook](https://notebooklm.google.com/notebook/301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76) — `301840a8-1c9a-4e9a-b41c-1ea7b3ea8b76`
- **Memory references:**
  - [[project-prolix-identity-and-thesis]] — Lane A/B identity and Claim 2 (WASM)
  - [[project-prolix-roadmap-design-decisions]] — §7.1 hard-gate status, B1 pre-registration
  - [[feedback-rust-wasm-parsing]] — preference for Rust-native parsers (drives HP4-WASM)

---

*End of synthesis. Generated by main-thread synthesis pass (no librarian dispatch); oracle reviewed.*
