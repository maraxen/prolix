Vendored OpenMM 8.3.1 Reference gold for campaign `nonbonded-omm-parity`.

Emit (Reference platform only):

    uv run --extra openmm python scripts/experiments/nl_omm_parity.py --emit-oracle --probe water --out outputs/nonbonded_omm_parity/emit_nl.json
    uv run --extra openmm python scripts/experiments/flash_omm_parity.py --emit-oracle --probe water --out outputs/nonbonded_omm_parity/emit_flash.json

Commit `nl_water.json` / `flash_water.json` plus this README. Record sha256 in
`MANIFEST` after emit. Do not use CUDA OpenMM as the energy oracle.

NL water gold: two LJ particles, pair at 8.5 Å (inside the OpenMM switch window),
cutoff 9 Å, switch 8 Å, CutoffPeriodic, 20 Å cubic box.

Flash water gold: same geometry, cutoff = MIC diameter (10 Å for the 20 Å box),
no switching.
