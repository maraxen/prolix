Vendored OpenMM 8.3.1 Reference gold for campaign `nonbonded-omm-parity`.

Emit (Reference platform only):

    uv run --extra openmm python scripts/experiments/nl_omm_parity.py --emit-oracle --probe water --out outputs/nonbonded_omm_parity/emit_nl.json
    uv run --extra openmm python scripts/experiments/flash_omm_parity.py --emit-oracle --probe water --out outputs/nonbonded_omm_parity/emit_flash.json
    uv run --extra openmm python scripts/experiments/nl_omm_parity.py --emit-oracle --probe 1vii --out outputs/nonbonded_omm_parity/emit_nl_1vii.json

Commit `nl_water.json` / `flash_water.json` / `nl_1vii.json` plus this README.
Record sha256 in `MANIFEST` after emit. Do not use CUDA OpenMM as the energy
oracle.

NL water gold: two LJ particles, pair at 8.5 Å (inside the OpenMM switch window),
cutoff 9 Å, switch 8 Å, CutoffPeriodic, 20 Å cubic box.

Flash water gold: same geometry, cutoff = MIC diameter (10 Å for the 20 Å box),
no switching.

NL 1vii gold: solvated 1VII (ff19SB + amber14/tip3p water, addHydrogens +
addSolvent padding=0.8 nm), OpenMM Reference PME + NL cutoff/switch matching
`REGRESSION_EXPLICIT_PME` (cutoff/alpha/grid), switch = cutoff - 1.0 Å,
dispersion correction off. Vendored energy/forces/positions/box/charges/LJ
params for later `energy_fn_from_bundle`/EnsemblePlan comparison — the
prolix-side comparison itself (backlog #4383) is not yet implemented;
`nl_omm_parity.py`'s `_compare_probe` currently returns a deliberate
gate_pass=0 stub for probe="1vii" pending that wiring.
