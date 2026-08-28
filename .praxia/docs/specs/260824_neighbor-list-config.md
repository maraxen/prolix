---
title: Neighbor-list config + flash tiles + vendored OpenMM parity
date: '260824'
status: in-progress
---

# NeighborList / NonbondedConfig and flash tile body

Pallas PME is deferred (ADR-006). This spec locks the OpenMM-shaped NL contract, flash JAX mitigations, and the vendored OpenMM 8.3.1 Reference bathos campaign.

## Engines (mutex)

`NonbondedEngine`: `dense` | `neighbor_list` | `flash`. Flash is tiled all-pairs, not a cutoff list. Do not default-flip from a timing race (lesson 379).

## NeighborListConfig (v1)

| Field | Default (new API) | Legacy flags path |
|-------|-------------------|-------------------|
| cutoff | bundle cutoff | 9.0 fallback |
| skin | 1.0 Å | 0.5 Å (preserve jax-md `dr_threshold`) |
| switch_width | None (off) until campaign pass | None |
| capacity_multiplier | 1.25 | same |
| max_exclusions | 32, **raise** if topology needs more | raise |
| nl_update_every | None = Verlet only | 20 AND Verlet |

Rebuild: `max|Δr| > skin/2` OR optional stride. LJ switch: OpenMM 8.3 polynomial on `[rcut-width, rcut]`. No Coulomb switch. Export/WASM: NL not on jax.export.

## Flash (appendix)

Vectorize exclusion scales; fuse into pair weights; keep T=256 remat-on; pad implicit `N//T` tail. No OpenMM switch on flash (all-pairs).

## Campaign `nonbonded-omm-parity`

Gold: `data/oracles/openmm_8.3.1/`. Emit with OpenMM 8.3.1 **Reference**. Parity vs gold. Flash gold uses MIC-diameter cutoff, not 9 Å. Frozen `xr_parity_omm_tip3p` sidecars are not edited.
