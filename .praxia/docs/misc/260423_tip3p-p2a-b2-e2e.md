# P2a-B2 end-to-end (non-blocking)

Full Tier‑1 `tip3p_ke_compare.py` runs are **OpenMM-heavy** and belong on **Slurm / nightly** runners, not default PR CI.

## Command (primary profile, `openmm_ref_linear_com_on`)

```bash
mkdir -p outputs/logs/local_tip3p_ke

OPENMM_INSTALL_MODE=ephemeral uv run --with openmm python scripts/benchmarks/tip3p_ke_compare.py \
  --engine both --jax-x64 on --n-waters 33 --steps 30000 --burn 10000 --sample-every 10 --replicas 5 \
  --timing-mode both --warmup-steps 100 --measure-steps 500 \
  --remove-cmmotion true --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0 --openmm-integrator middle \
  > outputs/logs/local_tip3p_ke/tier1_both_openmm_ref_linear_com_on_x64_r5.json
```

## Gates on the artifact

```bash
uv run python scripts/benchmarks/tip3p_ke_gates.py \
  outputs/logs/local_tip3p_ke/tier1_both_openmm_ref_linear_com_on_x64_r5.json
```

Exit code `0` means **P2a-B2-R both** passed; JSON stdout includes **P2a-B2-X** (G4) pass/fail for logging.

## Long-run tightening aggregate (same gates)

After Slurm/array runs of `scripts/benchmarks/tip3p_langevin_tightening.py`, combine tee logs:

```bash
uv run python scripts/benchmarks/aggregate_tip3p_tightening_logs.py \
  --from-dir outputs/logs/tip3p_tightening_run --require-profile-id \
  > outputs/logs/tip3p_tightening_aggregate.json
```

The stdout JSON carries **`schema`: `tip3p_tightening_aggregate/v1`**. Evaluate with the **same** gate CLI:

```bash
uv run python scripts/benchmarks/tip3p_ke_gates.py outputs/logs/tip3p_tightening_aggregate.json
```

## Slurm

Point your array or single task at the same command line; record **job ID**, **commit SHA**, and **output path** in the PR or memo that consumes the artifact.
