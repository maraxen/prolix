---
title: "DRAFT — xtrax `jax-profiling` skill (scoped from prolix P1–P8)"
status: draft
date: '260820'
parent_spec: 260817_jax-profiling-optimization-workflow
jax_profiling_contract_version: "3.0"
not_p9: true
---

# DRAFT: `jax-profiling` skill for xtrax

**This is not P9.** It does not live under `~/projects/xtrax/agent_assets/skills/`.
It is not installed by `just install-skills`. Do not cite it as the skill.
Do not open an xtrax PR from this file until the open sections below are filled
from a real Stage-2 pull → report → hunt decision.

**Why a draft now.** The parent spec sequenced P9 last so the skill would not
encode guesses that P5/P6/P7 had to resolve empirically. Those steps have
enough *measurement* lessons to freeze the TIER-1 judgments. They do **not**
yet have the *use* lessons: what a ranking over real GPU records looks like,
which scopes are trusted, and what we decide to change in the kernel. Those
go in §Open after the pull.

**Eventual home (P9, later):**

```
~/projects/xtrax/agent_assets/skills/jax-profiling/SKILL.md
~/projects/xtrax/agent_assets/skills/jax-profiling/references/{stages,instrumentation,claims,loops,failure-modes}.md
```

Layout must match `using-xtrax` (YAML frontmatter, TIER-1 self-contained,
TIER-2 one file per topic, Workflow Index, backlink line on every reference).
Work on an xtrax *branch*; never commit this to xtrax `main` from a prolix session.

**xtrax ships no profiling API.** `ProbeRecord` / `assert_claim_supported` are
a copyable pattern. A worked implementation exists in prolix
`scripts/profiling/` (`CONTRACT_VERSION = "3.0"`). Citing that path in the
skill is a labelled example, never an instruction.

Worked examples from this campaign are marked **(prolix, example)** so they
cannot be copied as JAX-general rules.

---

## Intended frontmatter (P9)

```yaml
name: jax-profiling
description: >
  Staged JAX performance measurement and a machine-checked claim-class
  contract. Use when asking why JAX code is slow, whether a number may be
  cited, or which probe answers which question. Not for correctness or NaNs.
jax_profiling_contract_version: "3.0"
triggers:
  - why is this JAX code slow
  - failed throughput gate
  - suspected recompilation / dispatch regression
  - can I cite this profiling number
  - cost_analysis
  - named_scope
  - jax.profiler.trace
  - ProbeRecord
  - ClaimClass
  - assert_claim_supported
  - TERM_RANKING
```

If a record in hand stamps a different **MAJOR** `contract_version` than this
pin, distrust the claim rules here and re-read the repo's `claims.py`.

---

## TIER-1 — `SKILL.md` body (draft)

### When to use / when not to

Use for performance questions only: where time goes, whether a compile/dispatch
count regressed, whether a term ranking is legal to cite, whether an end-to-end
number may be extrapolated.

Do **not** use for correctness, NaNs, energy drift, or thermostat bugs. Those
are a different loop; a fast wrong kernel is still wrong.

The harness is not the speedup. A complete Stage-0..2 pipeline with no ranking
read and no kernel change has not started the hunt. **(prolix, example:** P1–P8
built the contract; accelerating the MD kernel starts at pull → `report.py` →
decision.)

### The four stages

| Stage | What it is | Can prove | Cannot prove |
|---|---|---|---|
| 0 | Free structural (`cost_analysis`, HLO shape, dispatch *structure*) | "this program has N compiled modules / this fusion exists" | Any wall-clock ranking of terms; any GPU claim |
| 1 | Cheap CPU execution (dispatch counts, bonded-scale micros) | Compile/trace/execution *counts*; CPU-only micros | Term ranking on GPU; end-to-end at production `n_atoms` |
| 2 | GPU, small system, attributed scopes | Term ranking **on that device, at that size, under those XLA flags** | Production-scale ranking; CPU→GPU transfer of the ranking |
| 3 | GPU, target-scale end-to-end | End-to-end at the stamped `n_atoms` | Nothing below its own `n_atoms` if you then extrapolate past the scale limit |

Once Stage 2 exists, Stage 0/1 demote to **regression tripwires**. They never
block a ranking gate and are never cited for `TERM_RANKING`.

### Roofline rule (why TERM_RANKING is stage ≥ 2)

FFT-like work is bandwidth-bound (~0.3–0.5 FLOPs/byte). Dense GEMM-like work is
compute-bound (~1.5–3.5+). Those two reorder between CPU and GPU. A CPU-derived
"PME is cheap / nonbonded is the bottleneck" ranking is **structurally invalid**
on GPU, not merely noisy. `assert_claim_supported(..., TERM_RANKING)` **raises**
if any selected source is stage < 2. Do not warn. Do not loosen the floor.

### Scale rule

Platform validity ≠ scale validity. Dense paths OOM or change memory regime
long before a linear `n_atoms` ratio looks scary. `END_TO_END` must declare
`target_n_atoms` (or explicitly opt out). `target / min(source n_atoms)` must
not exceed **10×**. That 10× is a cheap linear proxy, not a derived memory
bound: a false pass is possible if the real regime change is a memory
transition inside the ratio. Response to a raise: smaller target, or a larger
source — **never raise the limit to make it pass**.

### Provenance unanimity

`TERM_RANKING` and `END_TO_END` sources must agree on `x64_enabled`,
`xla_flags`, `device_kind`, `platform`, `git_sha`, and contract MAJOR.
Unverifiable `git_sha` (`unknown`, `*-dirty`, `*-unverified`) is not citable.

**(prolix, example:** mixed `XLA_FLAGS` alone produced a ~1170× throughput
change on Blackwell nodes. Cluster checkouts without `.git` need an explicit
SHA stamp — env / sidecar file — or every citable claim fails closed.)

### Absent scopes are `null`, never `0.0`

A label that was expected but did not appear in the executed trace is `None`.
Zero exclusive time means the scope ran and was empty. Coercing `null` → `0.0`
invents a ranking term.

A ranking that mixes `named_scope` and `op_name` attribution is reportable but
**must be labelled mixed** (banner + per-row method). Empty
`attribution_method` is only legal when every scope value is `None`.

### Minimal loop (pattern, not a library)

1. Emit a stamped record (stage, `n_atoms`, platform, device, sha, flags, metrics, scopes).
2. `assert_claim_supported(records, claim)` **before** any ranked table or paper sentence.
3. On `ClaimValidityError`: get a better record, or weaken the claim. Never loosen the guard.

`select_sources` is part of the contract: the caller hands over everything
loaded; the contract picks what backs the claim. Pre-filtering to launder a
CPU record into a ranking is the failure mode the raise exists to catch.

`TERM_RANKING` additionally requires ≥2 non-`None` scopes on a selected source.
A Stage-2 record with `scopes is None` or a single labelled term cannot back a
ranking — capture a trace, not just `total_step_seconds`.

### Workflow Index (eventual)

1. Which stage / which tool → `references/stages.md`
2. Adding `named_scope` without lying → `references/instrumentation.md`
3. May I cite this number? → `references/claims.md` (SKILL.md already answers the TERM_RANKING floor)
4. Discovery vs confirmation; GPU budget → `references/loops.md`
5. Known ways this skill goes stale or over-builds → `references/failure-modes.md`

---

## TIER-2 — `references/stages.md` (draft)

Question → tool:

| Question | Stage | Tool |
|---|---|---|
| What did XLA compile? How many modules? | 0 | `jax.jit(...).lower(...).compiler_ir()` / `cost_analysis()` |
| Did compile or dispatch count regress? | 1 | execution counters (`n_executions`, `n_compilations`, `n_jit_traces`) |
| Where does GPU step time go among named terms? | 2 | `named_scope` + `jax.profiler.trace` + HLO `op_name` map |
| Does the production-size step get faster if we change X? | 3 | wall-clock A/B at target `n_atoms` |

**Two-input attribution, not trace-only.** Executed profiler events carry
`hlo_op`. Scope paths live in compiled HLO metadata
(`op_name="jit(fn)/scope1/scope2/.../leaf"`). Build
`instruction-name → scope-path` from HLO (follow `calls=` / `to_apply=` to a
callee ROOT when the top-level thunk has no metadata), then look up each trace
event. Under `vmap` / `jvp`, path segments look like `vmap(jvp(label))` —
unwrap `word(...)` wrappers before matching `known_labels`.

**HLO presence ≠ executed thunk.** A `named_scope` can survive in compiled HLO
and still never appear as a top-level executed thunk: fusion absorbs it into a
larger block whose representative `op_name` is some other instruction.
Increasing loop trip count does not fix that (fusion is a compiler decision).
**(prolix, example:** `settle_*` labels present in HLO, absent from the
executed trace; `dense_bonded_*` recovered. Treat settle/PME wall-clock claims
as untrusted until attribution survives thunk-level matching, or fall back to
`op_name` and label the ranking mixed.)

**Config cells can emit Stage-2 records that still cannot rank.** GPU +
`total_step_seconds` is not enough. If the trace recovered 0 named-scope
labels, that cell is coverage for *presence*, not for term ranking.
**(prolix, example:** dense / PME-off cells often 0 labels; flash + PME-on
cells carried named-scope ranking.)

**GPU command buffers swallow named_scope.** On GPU, XLA often records a
CUDA graph as one executed thunk (`hlo_op=command_buffer_N`).
`parse_scopes` only keeps thunks that map to `known_labels`, so labelled
PME/FFT scopes can be a few percent of host `total_step_seconds` while the
graph holds the rest. Rank **all** `hlo_op`s first (`parse_hlo_op_times`).
If the top row is `command_buffer_*`, do not add more `named_scope`s — they
will not appear as top-level thunks. Diagnostic replay with
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` (empty) explodes the graph into
ordinary fusions; that is a **different program** (P5). Compare host step
on vs off before citing the exploded ranking. Profiler start/stop slices in
the same Perfetto file are tens of ms of host noise and are not the step.

**(prolix, example 2026-08-20:** `1vii` flash PME-on 1.0 Å, H200. Host step
1.076 ms. Sum of `hlo_op` exclusive time 1.012 ms. `command_buffer_14`
0.900 ms (320 hits). Named-scope PME labels ~23 µs. Dump:
`outputs/profiling/stage2/hlo_op_dump_1vii_flash_pmeon_g1.0.json`.)

---

## TIER-2 — `references/instrumentation.md` (draft)

### Adding `named_scope`

Instrument the fused region you will cite, not a helper that inlines away.
Do not split a fused `jax.checkpoint` tile body to make LJ vs Coulomb
separable: that is a logic change, perturbs fusion, and invalidates the
A/B this section exists for. Intra-tile splits are `op_name` only, or a
separately scoped restructuring with its own perturbation gate.

**Command buffers (GPU).** If an all-`hlo_op` dump is dominated by
`command_buffer_*`, extra `named_scope` will not surface inside the CUDA
graph. Empty `--xla_gpu_enable_command_buffer=` is a diagnostic compile,
not production. Compare host step; if it moves, the ranking is about
capture, not the math.

### Mandatory perturbation check

A/B total step time, scopes on vs off. Off-arm: patch `jax.named_scope` to a
null contextmanager **in a subprocess before importing the library under
test**. No env var, no permanent hot-path toggle.

- Force bitwise equality is the wrong gate. Fusion changes FMA order;
  `allclose` on forces (rtol/atol from the record's `x64_enabled`) is the
  semantic check. Nonzero `max_abs_force_delta` inside tolerance is reportable
  fusion change, not an automatic fail.
- Timing: median pairwise relative delta with a 90% percentile-bootstrap CI
  inside ±5%. ≥21 interleaved A/B pairs after a compile-absorbing warmup
  **inside each child**. Compile time must not enter `d_i`.

Two failure modes, do not collapse them:

| Verdict | Meaning | Action |
|---|---|---|
| DETECTED (`|median(d_i)| > 0.05`) | Scopes changed the hot path | Revert that scope *group* (bisect), record the fallback |
| INSUFFICIENT POWER (CI misses the margin, `|median| ≤ 0.05`) | Not enough pairs | Increase `n_pairs` from a pilot IQR; **do not revert on this branch alone** |

A clean CPU / small-N perturbation check **does not transfer** to GPU at
Stage-2 size. Replicate on the device you will rank, or record per-scope
attribution as unavailable and rank `op_name` only.

**(prolix, example:** Stage-2 replication on H200 for one flash+PME-on cell
passed with `n_pairs=21`, median delta ~8e-4, CI inside ±0.05. That licenses
named-scope ranking for *that* config, not for unlabeled cells.)

### Inference `while_loop` programs

Carry-only step loops (`lax.while_loop` / xtrax `WhileCarry`) are the
throughput path. They are not reverse-mode AD safe. Profile the jitted
inference program you actually run, not the scan/trajectory variant used for
AD. Nested `jit` under `vmap` is wrong; host call sites want one program.

---

## TIER-2 — `references/claims.md` (draft)

`ClaimClass`: `STRUCTURAL`, `DISPATCH_COUNT`, `TERM_RANKING`, `END_TO_END`.

Required metrics (contract 3.0):

- `STRUCTURAL`: none
- `DISPATCH_COUNT`: `n_executions`, `n_compilations`, `n_jit_traces`
- `TERM_RANKING` / `END_TO_END`: `total_step_seconds` (+ ranking needs ≥2 scopes)

**MAJOR bump trigger:** removing a required metric newly *accepts* previously
rejected claims. **(prolix, example:** `n_host_syncs` was specified, then
dropped — on CPU, device buffers *are* host memory, so a host-sync counter
cannot exist; shipping it would fail every Stage-1 record forever. 2.0 → 3.0.
Re-emit records after a MAJOR bump.)

Guards (a)–(d) as in TIER-1. `select_sources` is metric-keyed then
stage-maximal. Mixed contract MAJORs cannot share a claim.

A report renderer that only raises the contract without emitting a table is
not a report. Columns and mixed-attribution labelling are part of the claim
surface. Footer: `contract_version`, `git_sha`, `xla_flags` the sources were
unanimous on.

Legitimate responses to `ClaimValidityError`: better record, or weaker claim.
Illegitimate: raise `SCALE_EXTRAPOLATION_LIMIT`, drop unanimity, pre-filter
CPU records, coerce absent scopes to 0.

---

## TIER-2 — `references/loops.md` (draft)

**Discovery** is iterative and ungated (`scripts/explore/` in repos that have
that split). **Confirmation** is exactly one pre-registered experiment per
surviving hypothesis, with an anchored threshold (e.g. "removing X yields
≥1.5× on this path"). Do not put discovery in the confirmation tool.

Order of a hunt, once Stage 2 exists:

1. Pull records to the local tree (the reporter must not read a remote path).
2. `assert_claim_supported` then rank.
3. Read the table: hottest *trusted* scopes (named_scope, perturbation-clean,
   actually present as executed thunks).
4. Form a hypothesis. Explore ungated.
5. Promote survivors. Change the kernel. Confirm.

Cluster discipline:

- Check **live** contention (`scontrol`, current `gres`) before trusting a
  partition's reputation. **(prolix, example:** dedicated H200 partition had 2
  GPUs fully allocated for days; shared preemptable pool had ~194. Submit
  wrappers that inject a default CPU/GPU partition will silently miss the
  device you asked for — pass partition explicitly.)
- Sweep many hypotheses per GPU submission; small system before large.
- STEP gate (machinery demonstrated on one full pair) ≠ SIDECAR gate
  (enough *cells* to rank the design). Partial coverage ranks only covered
  cells and names the rest. <2 covered cells: no ranking.
- Frozen experiment sidecars: do not edit a registered sidecar to make a GPU
  task parse; use an explicit ungated path for probes that cannot satisfy it.

**§Open — fill after pull / emission / decision**

Pull started 2026-08-20 from worktree `wt-20260807-132628`.

Commands (JSON only; profiler traces left on cluster, ~468M under `stage2/`):

```bash
rsync -azi \
  engaging:/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628/outputs/profiling/stage2/*.json \
  outputs/profiling/stage2/
rsync -azi --exclude='*_trace' --exclude='*_trace/**' \
  engaging:/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628/outputs/profiling/stage2/scope_ab/ \
  outputs/profiling/stage2/scope_ab/
```

`bth sync engaging --pull` failed (`rsync` 23). Catalog not refreshed. Cluster
`coverage.json` reports `n_configs_with_valid_dense_flash_pair: 0` for all six
cells despite 12 ProbeRecords — SIDECAR join is not a ranking license.

- `TERM_RANKING` after restamp: **passes** on the four flash+PME-on records (`git_sha=e7846e7…`, `xla_flags=''`). Table: `outputs/profiling/stage2/p8_report.md`.
  Attribution is `named_scope` only. Hottest *labelled* scopes are `pme_fft_inverse` then `pme_fft_forward` then `pme_greens_setup`; `flash_nonbonded_tiles` appears only on 1vii (~1% of attributed time). Most known labels are `absent` (not in the executed thunk). Exclusive labelled times are ~µs vs `total_step_seconds` ~1 ms — the ranking is among surviving PME FFT scopes, not a full-step breakdown.
- Restamp (not `83cc282`): array JSON predates that commit (2026-08-20 07:35 PDT). Logs never printed `PROLIX_GIT_SHA`. Chain of custody:
  - Four flash+PME-on files: array **20762713** tasks 4/5/10/11 COMPLETED; JSON `flash_forces_ms` matches `20762713_4.out`; submitted after `e7846e7` push, before `ded45ed`. SHA **`e7846e783e08709741cef26276cb016baf3094b1`**.
  - Other eight: array **20791998** (after `ded45ed` emit fix). SHA **`ded45edba40407f481938deaed45f9dedb609876`**. Unrankable (0 labels); not in the TERM_RANKING source set.
- Kernel decision: **not started.** Named-scope PME-FFT is not the 1 ms.
  Op-name dump of the existing GPU gzip: `command_buffer_14` is 0.900 ms of
  1.012 ms `hlo_op` time. Diagnostic replay with command buffers off is
  `scripts/slurm/prof_stage2_cmdbuf_off.slurm` (1vii flash PME-on 1.0 Å).
  Submitted 2026-08-20: SLURM **20846317** COMPLETED 00:01:36 node3001.
  Host step **1.120 ms** vs cmdbuf-on **1.076 ms** (~+4%; capture was hiding
  the millisecond, not creating it). `hlo_op` dump has **no**
  `command_buffer_*`, 44 distinct ops, sum 0.988 ms. Hottest:
  `loop_multiply_fusion` 0.261 ms × **20** (padded 5000 / tile 256).
  Named-scope still ~75 µs (`pme_bwd_gather` newly visible at 46 µs).
  Do not mix this record into the e7846e7 TERM_RANKING set (`xla_flags` and
  `git_sha` differ).

  **Mapped (cmdbuf-off Perfetto, 2026-08-20 gzip, no extra HLO dump):**
  Pad is `ceil(5000/256)*256 = 5120`. Every hot thunk except PME/FFT has
  **n=20** = `lax.scan` over tiles. Sibling kernels that *do* carry
  `args["name"]` all say
  `jit(flash_forces)/flash_grad/transpose(jvp(flash_nonbonded_tiles))/while/body/closed_call`.
  `loop_multiply_fusion` itself has empty `name` (fusion); grid `2560×128`
  = 327680 threads = `(5120×256)/4` pair-elements — the checkpointed
  `_compute_tile_inner` elementwise body (LJ + `erfc` Coulomb). Occupancy
  43.75%, 69 regs. `loop_select_fusion` same grid = `jnp.where(pair_mask)`.
  `copy.5`/`copy.7`: D2D **5,242,880 B** = `5120×256×4` f32 `(N,T)` tiles
  (scale / pair buffers; remat copies). `input_scatter_fusion*`: grid 64×128
  = 8192 = `T × max_excl` (256×32) — sparse
  `vdw_scale_tile.at[...].set`, **not** PME spread. PME is the **n=1**
  kernels (`jvp(pme_bwd_gather)`, FFTs). Kernel target is the tiled
  pair-energy fusion + `(N,T)` traffic, not FFT.

- Tile × remat discovery (not TERM_RANKING): SLURM **20850993** array 0–5,
  1vii flash PME-on 1.0 Å, cmdbuf **on**, bonded+flash host timer.
  Control T=256 remat=on = **1.089 ms** (vs P7 1.076 ms, +1.2%, composition OK).
  Remat off is ~3.5–4× slower at every T. T=512 remat on = 0.913 ms =
  **1.19×** vs control — under the ≥1.2× promote bar. Leave production
  `T=256` remat on. Table: `outputs/profiling/stage2/tile_checkpoint/table.md`.

Remaining pull: traces only if we need `trace.py` ground-truth on one GPU
trace; bathos catalog via fallback rsync if SIDECAR must be re-derived.

---

## TIER-2 — `references/failure-modes.md` (draft)

Each entry has a **tell**.

1. **Bottleneck-found-in-week-one over-build**
   Tell: weeks of harness, fixtures, and skill prose; no ranking read; no
   kernel diff. The live throughput gap is untouched.

2. **Unrepresentative probe scale**
   Tell: ranking cited at production `n_atoms` from a probe whose ratio
   exceeds 10×, or from CPU Stage 1. Dense OOM / memory-regime change is the
   pre-mortem; the linear 10× guard is a proxy.

3. **Skill rot / false authority**
   Tell: `jax_profiling_contract_version` MAJOR behind the records in hand;
   one enforced raise still passes while unenforced judgments (partition
   folklore, "named_scope always attributes", "CPU perturbation transfers")
   have drifted. Mitigation is a follow-up PR to the skill on a MAJOR bump
   — unmitigated by CI because the skill lives in a repo with no dependency
   on the measurement code.

4. **HLO-labelled, thunk-invisible scopes**
   Tell: exclusive time claimed for a label that appears in `--hlo-only`
   counts but not in `parse_scopes` on the executed trace.

4b. **Command-buffer (CUDA graph) as the missing millisecond**
   Tell: named_scope ranking is µs; host `total_step_seconds` is ~ms; an
   all-`hlo_op` dump is dominated by `command_buffer_*`. Treating labelled
   FFT as the kernel bottleneck. Fix: dump `hlo_op`s; diagnostic
   command-buffer-off replay; do not restamp the production ranking from
   the off capture.

5. **Fusion perturbation laundered as instrumentation**
   Tell: scopes-on vs off fails DETECTED and the ranking is still cited as
   named_scope; or INSUFFICIENT POWER treated as revert.

6. **Coverage/presence confused with ranking**
   Tell: "we have 12 GPU JSON files" used as permission to publish a term
   table when half the cells have `scopes` all-`None`.

7. **Unverifiable provenance**
   Tell: `git_sha=unknown` or dirty; mixed `xla_flags`; ranking still shipped.

8. **Remote-path reporter**
   Tell: analysis step SSHs or reads cluster paths. Pull first.

9. **Nested-jit profile of the wrong program**
   Tell: scan/AD trajectory profiled; production path is `while_loop` /
   `WhileCarry`.

---

## Open questions (do not encode as guidance until answered)

- [x] First real `TERM_RANKING` over pulled Stage-2 records: **blocked** (`git_sha=unknown` on all 12 array ProbeRecords). Scope-count preview (not a citation): flash+PME-on cells only.
- [x] Attribution on those four cells: `named_scope` only. Eight cells have zero executed labels.
- [ ] Hottest term survive thunk-level matching vs HLO-only? Traces not pulled.
- [ ] What kernel change is the first confirmed experiment? Blocked on a citable ranking.
- [ ] xtrax `.praxia/manifest.toml`: does `jax-profiling` need `[[plugin.skills]]`
      (`using-xtrax` has none today)? Resolve at P9 PR time, state in the PR.
- [ ] After two clean hunt loops, revisit S8 (upstream `scripts/profiling/`
      *code*). Landing this skill does **not** fire S8.

---

## Split map for P9

| This draft section | P9 file |
|---|---|
| Intended frontmatter + TIER-1 | `SKILL.md` |
| TIER-2 stages | `references/stages.md` |
| TIER-2 instrumentation | `references/instrumentation.md` |
| TIER-2 claims | `references/claims.md` |
| TIER-2 loops + §Open | `references/loops.md` |
| TIER-2 failure-modes | `references/failure-modes.md` |

When splitting: every reference file starts with

```
> Part of the `jax-profiling` skill (`agent_assets/skills/jax-profiling/SKILL.md`) — TIER-2 deep reference.
```

Strip **(prolix, example)** blocks out of the instruction voice; keep at most
one labelled worked example per rule if it still earns its place.
