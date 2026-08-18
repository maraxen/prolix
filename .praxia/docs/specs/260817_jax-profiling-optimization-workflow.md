---
title: JAX Profiling & Optimization Workflow — Architecture Decision + Phase-2 Plan
description: Layered split (jax-profiling skill for judgment, authored cross-repo in xtrax/agent_assets/skills/ + prolix-local scripts/profiling/ primitives + machine-checked stage/scale validity contract), with bathos scoped to the confirmation loop and the xtrax CODE upstream still deferred behind a named trigger; plus an ordered CPU-first implementation plan ending in one small-system H200 probe and one cross-repo skill-authoring step
status: proposed
task_id: 260817_jax-profiling-optimization-workflow
date: '260817'
backlog_ids: ''
adversarial_review: ''
---

# JAX Profiling & Optimization Workflow

Derived from brainstorm session `56d2904e`
(`.praxia/docs/specs/260817_where-should-a-reusable-staged-cpu-synth.md`), which reached
VERDICT: PASS on the layered split S3 + S7 + S8 + S9 with a pre-mortem scale amendment, and
an accepted INVEST-Small narrowing that §0 overrides for one step with stated grounds.

## 0. Terminology note (resolves a naming collision)

The brainstorm numbers its rollout by *measurement stage* (Phase 1 = CPU instrumentation +
harness; Phase 2 = GPU small-system; Phase 3 = DHFR end-to-end). The dispatching workflow
calls this document's implementation plan "phase 2" meaning *the implementation phase that
follows this design phase*. To avoid ambiguity, this spec uses:

- **Stage 0/1/2/3** — the measurement-validity axis (fixed, non-negotiable).
- **Steps P1..P9** — the ordered implementation units below. P1–P8 land in this repo; P9 is
  the single cross-repo step (`~/projects/xtrax`) and is flagged as such at its own heading.

Steps P1–P6 are the brainstorm's Phase 1 (all free/local/CPU). Step P7 is the **single**
GPU step in scope: the opening move of the brainstorm's Phase 2, small system only.
DHFR-scale (Stage 3) work is explicitly **out of scope** here.

**Stated override of an ACCEPTED brainstorm decision.** The brainstorm's final Decision Log
entry ([ACCEPT] INVEST "S") narrows the committed spec to Phase 1 only, with Phase 2
recorded as a follow-on, on the ground that this hedges pre-mortem story 1 (over-build).
This spec deliberately overrides that for exactly one step, P7, and for one reason: Phase 1
alone cannot answer any term-ranking question — the contract it builds *forbids* it by
construction, so a Phase-1-only spec ships an enforcement mechanism and zero enforceable
conclusions, and the live RR5 blocker is untouched. Including the single smallest Stage-2
probe is what makes the Phase-1 machinery load-bearing rather than speculative, which serves
the same anti-over-build rationale the decision invoked. The override is bounded: one step
(P7), one small system, no DHFR. §5's in-repo acceptance list has 14 criteria across P1–P8,
with a separate cross-repo subsection for P9; the INVEST-Small concern the brainstorm raised
is addressed by P7's scope being one array submission on one small system, not by a criterion
count. Stage 3 and hypothesis promotion remain follow-on rows in §4.

---

## 1. Architecture decision, restated

Profiling capability is split by *what each layer durably captures*, because the four
candidate homes capture four different things and picking one box loses the other three.
**A Claude Code skill (D) owns the judgment** — which stage answers which question, which
probe to reach for, when a result may be cited — because that judgment is prose-shaped and
has no natural code home. That skill lives in **xtrax**, at
`agent_assets/skills/jax-profiling/` (see the corrected boundary subsection below); the
measurement code does not follow it there. **A prolix-local `scripts/profiling/` module (B) owns the
measurement primitives**, written against a deliberately xtrax-shaped seam (pure functions
over trace/cost records, no `prolix` imports in the core), so the eventual upstream is a
move rather than a rewrite. **A machine-checked stage/scale validity contract (S7) is the
load-bearing artifact**: every probe emits a record stamped with `stage ∈ {0,1,2,3}` and
`n_atoms`, plus the claim classes it may support, and the analysis step *raises* rather than
warns when asked to draw a term-ranking conclusion from a sub-Stage-2 record or to
extrapolate end-to-end across more than ~1 order of magnitude of system size. Selection is
part of the contract, not the caller: the caller hands over everything it loaded and the
contract decides what backs the claim, so a caller cannot launder an invalid claim by
pre-filtering. This is the
one rule prose cannot enforce — the roofline argument (FFT is bandwidth-bound at ~0.3–0.5
FLOPs/byte, dense GEMM is compute-bound at 1.5–3.5+) means a CPU-derived "PME is cheap"
ranking is structurally invalid on GPU, and the pre-mortem showed the scale axis needs the
same guard independently, since dense OOMs at DHFR scale and the few-thousand-atom probe
therefore sits in a qualitatively different memory regime. **Bathos (C) is scoped to the
confirmation loop only** — discovery is iterative and belongs ungated in `scripts/explore/`;
each hypothesis that survives discovery is promoted to exactly one pre-registered bathos
experiment with an anchored threshold, which dissolves the "bathos can't track discovery"
gap rather than working around it (discovery was never in bathos's remit). **xtrax (A) is
deferred behind a named trigger** — upstream when a second consumer repo asks, or when the
API survives two complete bottleneck-hunt loops without a breaking change; absent a trigger,
upstreaming is *declined*, not silently pending. Rejected outright: A-first (puts a
cross-repo release cycle on a live blocker's critical path, and generalises an API we have
not yet used once), C-first (same critical-path objection plus it would encode a category
error into a cross-project tool), and B-only-disposable (fails the provenance criterion for
paper-citable numbers, and is empirically false in this repo — the `profile_b1_*` scripts
are still in use months later).

### Cross-repo boundary (explicit) — CORRECTED 2026-08-17

An earlier revision of this section asserted: *"No change in this plan lands in
`~/projects/xtrax`. That is a separate repo with its own release cadence and is not editable
from this worktree."* **Both halves are wrong.** `~/projects/xtrax` is a live local git repo
(its own `.git`) and is editable. And this plan *does* land a file there.

The error was structural, not incidental: it collapsed two separable things under the single
word "xtrax." Keep them apart, because they have different homes, different triggers, and
different review paths.

**The measurement primitives stay prolix-local — this part of the original reasoning is
unchanged and still correct.** `scripts/profiling/` (`record.py`, `claims.py`, `trace.py`,
`report.py`), the `named_scope` additions in `src/prolix/physics/flash_explicit.py` and
`src/prolix/batched_energy.py`, and every probe under `scripts/explore/`,
`scripts/experiments/`, and `scripts/slurm/` land in **this** repo. Whether that *code* is
ever upstreamed to `xtrax.profiling` remains governed by the S8 trigger (§4) — a second
consumer repo asks, or the API survives two complete hunt loops without a breaking change —
and absent a trigger it is declined, not pending. The seam discipline still binds: no file
under `scripts/profiling/` may import `prolix`.

**The skill does land in xtrax.** Confirmed path:

```
/home/marielle/projects/xtrax/agent_assets/skills/jax-profiling/SKILL.md
```

Not prolix, and not `~/.claude/skills/`. This is confirmed by direct filesystem inspection,
not inferred: xtrax already runs exactly this convention for an existing skill,
`agent_assets/skills/using-xtrax/SKILL.md` plus `.../using-xtrax/references/*.md`. §3 mirrors
that structure rather than inventing one. The placement is also the right one on the merits —
a staged JAX profiling methodology (roofline validity, stage/scale claim classes, tool
selection) is JAX-general, not molecular-dynamics-specific; filing it in prolix would make a
general methodology reachable only from one domain repo.

> **Do not read "the skill lands in xtrax" as "the S8 upstream trigger fired."** It did not.
> A prose agent asset has no import surface, no Python API, no version pin, and no alpha
> release: adding `agent_assets/skills/jax-profiling/` changes nothing about what prolix
> imports. Upstreaming `scripts/profiling/` *code* to `xtrax.profiling` is a separate,
> still-deferred decision. §4 keeps them as two rows for this reason.

Because the skill crosses a repo boundary, authoring it is a **cross-repo step (P9)** and is
called out as such in §2 — xtrax has its own review process, and editing another repo's files
is a real decision, not a formality.

**No change lands in the bathos tool itself.** Steps P1–P6 live in `scripts/explore/` and
`scripts/profiling/`, both ungated. Step P7 uses the existing sidecar *mechanism* — no new
bathos schema and no new campaign mode — but its own new sidecar and new exploration
campaign, because the existing one has historical runs and a scalar hypothesis that a
multi-config sweep cannot adjudicate.

---

## 2. Implementation plan (ordered, cheapest first)

Cost legend: **FREE** = local, no execution or trivially cheap CPU. **CHEAP** = local CPU,
executed, seconds-to-minutes, N ≤ 64 atoms. **GPU** = cluster submission. **CROSS-REPO** =
edits files outside this repo; cheap to *write*, but carries another project's review
process.

Every step is independently gate-able and independently revertable. One ordering constraint
overrides the cheapest-first heuristic: **no step may emit a `ProbeRecord` to
`outputs/profiling/` or commit a fixture under `tests/profiling/fixtures/records/` until P1's
Gate paragraph passes in full.** A record stamped `contract_version = "1.0"` by an
implementation that does not yet enforce 1.0's guards carries a version string that denotes
two different contracts, and no bump rule can repair that after the fact. P3, P4, P6 and P7
are therefore gated on P1's gate, not merely sequenced after it. This is verified cheap to
honour: at the time of writing no `ProbeRecord` has been emitted anywhere in the repo.

Steps P1–P8 land in this repo. **P9 is the one cross-repo step** (`~/projects/xtrax`) and is
flagged as such at its own heading — see §1.

---

### P1 — `scripts/profiling/` core module: stage+scale-stamped record and claim-class assert
**Cost: FREE.** No prolix imports; this is the xtrax-shaped seam.

Create:
- `scripts/profiling/__init__.py`
- `scripts/profiling/record.py` — `ProbeRecord` dataclass: `probe_id`, `stage: int`,
  `n_atoms: int`, `platform: str` (`cpu`/`gpu`), `device_kind: str | None`, `git_sha`,
  `timestamp`, `x64_enabled: bool`, `jax_version: str`, `jaxlib_version: str`,
  `xla_flags: str` (verbatim `os.environ.get("XLA_FLAGS", "")`), `metrics: dict[str, float]`,
  `scopes: dict[str, tuple[float, int] | None] | None` — per-scope exclusive seconds paired
  with occurrence count, matching `trace.py`'s return type exactly; `None` as a *value* means
  the label was expected but absent from the trace (never `0.0`), `None` as the *field* means
  the probe captured no trace at all. Serialisation writes a tuple as a two-element array and
  an absent label as `null`; deserialisation restores tuple vs `None`, never coercing `null`
  to `0.0`.
  `attribution_method: dict[str, str] | None` — per-scope, one of `"named_scope"` |
  `"op_name"`. Required whenever `scopes` is not `None`. A ranking whose entries mix methods
  is reportable but must be labelled as mixed.
  `config: dict[str, str]` (default empty) — the free-form config axes this record was
  produced under, string-valued so it can carry non-numeric identity (e.g.
  `{"protein": "1vii", "mode": "flash", "pme": "on", "grid_spacing": "1.0", "scopes": "on"}`).
  Distinct from `metrics`, which is float-only. `probe_id` remains an opaque label and is
  never parsed.
  `__post_init__` coerces every `metrics` value with `float(v)`, raising `ClaimValidityError`
  naming the key if the coercion fails. This is not defensive typing: every producer in this
  plan (P4 dispatch counters, P5 `ab_max_abs_force_delta` from a `jnp` reduction, P7
  `flash_speedup`) naturally hands over a numpy/jnp scalar, and `json.dumps` raises on those at
  `write()` time — after the measurement, on the cluster.
  Plus JSON (de)serialisation. Records are plain data —
  no JAX import at module scope. All of `x64_enabled`/`jax_version`/`jaxlib_version`/
  `xla_flags` are DEFAULTED FROM THE LIVE ENVIRONMENT at construction (`default_factory`), so
  a producer never has to remember them — a provenance field a caller can forget is a
  provenance field that will be forgotten. They remain settable because deserialisation
  requires it; deserialisation is the only sanctioned override path, and guard (d) therefore
  trusts the provenance the record carries. Deliberate forgery is out of the contract's threat
  model, which is accidental laundering by pre-filtering.
  `n_atoms: int` — the **real** (unpadded) atom count of the production system, never a
  padded count and never a reduced smoke-mode count. Where a probe pads for bucketing, the
  padded count SHOULD go in `metrics["n_padded_atoms"]`. This is best-effort documentation,
  deliberately not enforced: a padded count is meaningless for a non-bucketed probe, so
  requiring the key unconditionally would be wrong, and no contract guard reads it. The scale
  guard reasons about physical system size and keys exclusively on `n_atoms`, which is
  required, validated (`> 0`), and defined as the real unpadded production count — so omitting
  `n_padded_atoms` costs interpretability, never correctness of the guard. P6 stamps both (its
  `--dry-run` path already emits `n_real_atoms`/`n_padded_atoms`); other probes stamp it where
  they have it.
  `device_kind` is **auto-captured like the other provenance fields, not caller-declared**,
  because it is a guard-(d) unanimity field and a free-form caller-supplied string defeats the
  guard in both directions (two tasks on one machine spelling it differently spuriously
  invalidate a sweep; two tasks on different machines both hardcoding a literal silently pass
  one). `_capture_device_kind()` returns `_normalize_device_kind(jax.devices()[0].device_kind)`
  when a device is available and `None` otherwise (Stage 0 does not execute, and
  `platform="cpu"` records do not need it). `_normalize_device_kind` lowercases, strips a
  leading vendor token (`nvidia `, `amd `), and collapses internal whitespace to `_`, so
  `"NVIDIA H200"` → `"h200"` and `"NVIDIA RTX PRO 6000 Blackwell"` →
  `"rtx_pro_6000_blackwell"`. It is deliberately a normalisation, not an allow-list: an
  unrecognised device yields its own normalised spelling rather than an error, so a new device
  kind is recorded faithfully instead of blocking a probe. It remains settable for
  deserialisation, on the same terms as the other auto-captured fields. **Consequence for
  P7's gate:** read the `device_kind="h200"` assertion as `device_kind == "h200"` *after
  normalisation*, and P1's test suite includes `_normalize_device_kind("NVIDIA H200") ==
  "h200"`.
  Plus `__post_init__` validation, raising `ClaimValidityError` on any of:
  `stage not in {0,1,2,3}`; `stage >= 2 and platform != "gpu"` (the load-bearing
  implication — a Stage-2+ record is by definition GPU-measured, and `stage` is otherwise
  self-declared and unvalidated); `stage >= 2 and device_kind is None`; `n_atoms <= 0`.
  Stage is not derivable from platform in general (Stage 0/1 differ by execution, Stage 2/3
  by size), so this checks the one implication that is decidable rather than pretending to
  infer the field.
- `scripts/profiling/claims.py` — the contract. `ClaimClass` enum
  (`STRUCTURAL`, `DISPATCH_COUNT`, `TERM_RANKING`, `END_TO_END`);
  `permitted_claims(record) -> set[ClaimClass]` grants `STRUCTURAL` at `stage >= 0` and
  `DISPATCH_COUNT` at `stage >= 1` (Stage 0 does not execute, so it cannot count
  dispatches). Neither class raises on the stage axis beyond that floor, and that is
  deliberate: they are backend-dependent by nature, which is exactly why they are low-stage
  classes and why a CPU dispatch count is cited as a CPU dispatch count. The contract raises
  only where prose demonstrably fails — the term-ranking and scale rules.
  `select_sources(records, claim) -> list[ProbeRecord]` — returns the records that *back* a
  claim: the highest-stage records in the set that carry the metrics the claim ranges over.
  `assert_claim_supported` calls this internally; `records` is always the full loaded set,
  never a hand-filtered one.
  Selection is metric-keyed, then stage-maximal. `claims.py` defines
  `REQUIRED_METRICS: dict[ClaimClass, frozenset[str]] = {STRUCTURAL: frozenset(),
  DISPATCH_COUNT: frozenset({"n_executions","n_compilations","n_host_syncs","n_jit_traces"}),
  TERM_RANKING: frozenset({"total_step_seconds"}), END_TO_END:
  frozenset({"total_step_seconds"})}`. `select_sources(records, claim)` is: (1) keep records
  whose `metrics` keys are a superset of `REQUIRED_METRICS[claim]`; (2) for `TERM_RANKING`
  additionally require `scopes is not None` and contains ≥2 non-`None` entries — a ranking
  needs at least two terms to rank; (3) of what survives, return only those at `max(stage)`.
  An empty result is itself a raise (`ClaimValidityError`), and the message must name WHICH
  filter emptied the set: if the metric filter did, `no record carries the metrics this claim
  ranges over (required: [...])`; if the `TERM_RANKING` scopes filter did, `N record(s) carry
  the required metrics but none has >=2 non-None scopes entries — a ranking needs at least
  two terms to rank`. The two cases have different legitimate responses, so a message that
  conflates them is not actionable. The selector is fail-closed on both axes.
  `paired_configs(records, claim, *, axis: str, hold_fixed: tuple[str, ...]) ->
  list[tuple[ProbeRecord, ProbeRecord]]` — over `select_sources(records, claim)`, groups by
  the `hold_fixed` config keys, then within each group returns one `(a, b)` pair for every
  ordered-by-sorted-axis-value combination of distinct `axis` values — the full cross
  product within a group, not the groups themselves. A group with fewer than 2 distinct
  `axis` values contributes no pairs. This is the contract helper §P7 gates on;
  `select_sources` itself does not group. Raises `ClaimValidityError` if any selected source
  is missing the `axis` key or any `hold_fixed` key, naming the record and the key —
  defaulting a missing key to `""` would merge distinct cells and report a confounded pair as
  a pass.
  `assert_claim_supported(records, claim, *, target_n_atoms=None, allow_no_target=False)`
  which **raises `ClaimValidityError`** when (a) `claim is TERM_RANKING` and any record
  returned by `select_sources(records, claim)` has `stage < 2`; or (b) `claim is END_TO_END`
  and `target_n_atoms is None` and not `allow_no_target` — fail-closed: an end-to-end claim
  must declare either the system it extrapolates to or, explicitly, that it makes no
  extrapolation; or (c) `claim is END_TO_END` and
  `target_n_atoms / min(r.n_atoms for r in select_sources(records, claim)) >
  SCALE_EXTRAPOLATION_LIMIT` (default `10.0`) — `min`, not `max`, is deliberate: the claim is
  backed by *all* selected sources, so the guard asks whether every one of them is within the
  extrapolation limit — consistent with the contract being one-directional and fail-closed. A
  consequence, stated so it is not mistaken for a bug: adding a further valid same-stage
  record with a smaller `n_atoms` can tighten the guard and flip a passing claim to raising.
  The legitimate responses are the same two as always — narrow the claim to the sources that
  support it by declaring a smaller `target_n_atoms`, or obtain a larger-`n_atoms` source —
  never pre-filter the input set, and never raise `SCALE_EXTRAPOLATION_LIMIT`; or
  (d) `claim in {TERM_RANKING, END_TO_END}` and the selected source records are not unanimous
  in `x64_enabled`, `xla_flags`, `device_kind`, or `git_sha` (platform unanimity is already
  implied for `TERM_RANKING` by the stage≥2 floor plus the stage≥2 ⇒ platform=="gpu"
  construction check, and is checked explicitly for `END_TO_END`, which has no stage floor) —
  raises. Rationale: the cluster rules document `XLA_FLAGS=--xla_gpu_shard_autotuning=false`
  changing measured throughput by 1170× on `pi_so3`'s Blackwell nodes (node4007/node4008).
  P7 requests `--gres=gpu:h200:1`, which maps to node4009 (Hopper), where the repo documents
  the flag as a safe no-op — so the 1170× figure is the *evidence that XLA flags can reorder
  a measured ranking*, not a claim about P7's own device. The rule binds regardless:
  `xla_flags` is recorded verbatim, and a sweep mixing flagged and unflagged tasks is
  uncitable whatever the device. The project also benchmarks both precisions as a matter of
  standing discipline. `git_sha` is included because a term ranking assembled from a dense
  record at commit A and a flash record at commit B is not a ranking of the same code.
  `_capture_git_sha` appends `"-dirty"` when `git status --porcelain` is non-empty, and
  `select_sources` raises `ClaimValidityError` if any `TERM_RANKING`/`END_TO_END` source
  carries `git_sha == "unknown"` or a `"-dirty"` suffix — an unrecordable or uncommitted
  revision is not citable provenance. A ranking that silently mixes any of the four is not a
  ranking. Also
  `CONTRACT_VERSION: str` (MAJOR.MINOR,
  starts at `"1.0"`). **Bump rule:** major on any change that would newly reject a
  previously-accepted claim or newly accept a previously-rejected one (adding a rule,
  changing `SCALE_EXTRAPOLATION_LIMIT`, changing the stage floor); minor on additive changes
  that alter no verdict AND leave every previously-written record readable — a new
  `ClaimClass` member, or a new field declared with a plain default (e.g. `x: str | None =
  None`) and no `default_factory`. **A new field carrying a `default_factory` is a MAJOR
  bump, not a minor one**, and this is a direct consequence of read-side rule (ii) below:
  rule (ii) makes a missing `default_factory` key a hard raise, so adding such a field
  renders every record written before it unreadable — which alters every verdict those
  records backed, including P8's committed fixtures, every Stage-0/1 record on disk, and
  every Stage-2 record rsynced from the cluster. The rule is stated this way rather than
  softening rule (ii) because rule (ii) is the guard that stops a reading machine's
  environment being stamped into a measurement made elsewhere, and that protection is worth
  more than in-place readability of old records. **Consequence, stated so it is not
  discovered at read time:** a MAJOR bump invalidates all prior records for claim purposes.
  Records at the older MAJOR version are retained as history but are not citable and are not
  mixed into a source set (read-side rule (iii) enforces this). Regenerate P8's fixtures with
  `ProbeRecord.write` at the new version as part of the bump; re-emitting a cheap Stage-0/1
  record is minutes, and a Stage-2 record that cannot be re-emitted is recorded as lost to the
  bump rather than silently upgraded. `ProbeRecord` serialisation
  includes the `CONTRACT_VERSION` in force when the record was emitted.
  **Read-side contract.** `from_json` is fail-closed on both directions of version drift:
  (i) an unknown key raises `ClaimValidityError` naming the key and both contract versions;
  (ii) a MISSING key that has a `default_factory` raises `ClaimValidityError` rather than
  firing the factory — firing it would stamp the reading machine's environment into a record
  measured elsewhere, which is exactly the provenance corruption auto-defaulting exists to
  prevent. Deserialisation therefore requires every provenance field to be present.
  (iii) `select_sources` additionally raises if the selected sources disagree in the MAJOR
  component of `contract_version`; a minor disagreement is permitted, because by the bump
  rule a minor bump alters no verdict.
- `tests/profiling/test_claim_contract.py`

**Import mechanism (specified, because the repo has no convention and one of the obvious
fixes breaks the seam):** add `pythonpath = ["."]` to `[tool.pytest.ini_options]` in
`pyproject.toml`. Do **not** add a `sys.path.insert` inside the test, and do **not**
relocate `scripts/profiling/` under `src/prolix/` — the latter would place the seam module
inside the package it is forbidden to import from, making the no-prolix-import criterion
vacuous and converting the eventual xtrax upstream back into a rewrite.

**Gate:** `uv run pytest tests/profiling/ -q` passes, and the test suite includes at least:
a Stage-1 record raising on `TERM_RANKING`; a Stage-2 record permitting it; a Stage-2
record with `n_atoms=2000` raising on `END_TO_END` against `target_n_atoms=23558`; a
recursive, AST-asserted check that no file at any depth under `scripts/profiling/` imports
`prolix` (treating a relative `ImportFrom` whose `module` is `None` as unresolvable and
therefore a failure); a mixed set
of Stage-0, Stage-1 and Stage-2 records permits `TERM_RANKING` (the Stage-2 records are
selected), while the same set with the Stage-2 records removed raises — i.e. adding a valid
record must be able to *unblock* a claim, and adding an invalid one must never block one;
and an `END_TO_END` claim with `target_n_atoms` omitted and `allow_no_target` unset
**raises**; a record with `stage=2, platform="cpu"` raises at construction, not at
claim time; two Stage-2 records differing only in `xla_flags` raise on `TERM_RANKING`; two
Stage-2 records differing only in `device_kind` raise on `TERM_RANKING`; two Stage-2 records
differing only in `git_sha` raise on `TERM_RANKING`; a `git_sha` of `"unknown"` or carrying a
`"-dirty"` suffix raises on `TERM_RANKING`/`END_TO_END`; a Stage-2 record
with `scopes=None` raises on `TERM_RANKING` even though its stage qualifies, and the raise
message matches on the scopes reason, not merely the exception type; a Stage-2
record lacking `total_step_seconds` in `metrics` raises on `END_TO_END`; and adding a
further valid same-stage record with a smaller `n_atoms` to a passing `END_TO_END` claim's
source set flips the claim to raising — the `min`-reducer's deliberate consequence, pinned
so it is not mistaken for a regression; `paired_configs` returns the expected pairs on a
well-formed set, raises on a source missing a `hold_fixed` key, and returns no pair for a
group with a single `axis` value; a record constructed with a numpy `float32` metric value
round-trips through JSON; and a record missing a required provenance key on read raises
`ClaimValidityError` rather than firing its `default_factory`.
And `tests/profiling/` is collected under CI's marker expression
`-m "not (slow or integration or dynamics)"` — the grep-assert is worthless if CI never runs
it. Verify by checking the test appears in `tmp/pytest.json` after a CI-equivalent run.
**Scope: ~200 LOC.**

> Note the boundary is genuinely live: DHFR is 23,558 atoms, so a probe below ~2,356 atoms
> trips the assert and a mid-scale rung becomes mandatory. P6 measures the real number.

> **`SCALE_EXTRAPOLATION_LIMIT` is a deliberate cheap linear proxy, with a known false-pass
> case — stated here so it is not mistaken for a derived bound.** The regime change the
> guard exists to catch is a *memory* transition (dense pair storage is O(N²), PME cost
> tracks grid volume, NL cost tracks occupancy — none linear in `n_atoms`), and the true
> boundary is device- and precision-dependent, not ratio-dependent. A 10× `n_atoms` ratio is
> a ~100× dense-pair-memory ratio. Concretely: a 2,400-atom probe extrapolating to DHFR is
> 9.8× and **passes**, while crossing the dense-OOM transition. `n_atoms` is used anyway
> because it is the one quantity every probe can stamp without executing at the target size,
> and the guard is one-directional — it can only refuse claims, never approve one (approval
> still requires a Stage-2+ source record). **Consequence for P6:** a ratio in the 8–10 band
> is a *near miss*, not a pass; record whether the dense path fits in device memory at the
> probe size and again at the target, and treat a differing answer as triggering the
> mid-scale rung regardless of the ratio.

---

### P2 — `named_scope` coverage in the two uninstrumented hot paths
**Cost: FREE.** Required by every architecture option; lives in prolix source regardless of
where the harness lands. Prerequisite for any *term-attribution* claim, but run in parallel
with P1, not serialised behind it.

Create:
- `tests/physics/test_scope_inventory.py` — the home for the compiled-HLO scope inventory
  assertion below. It lives in `tests/physics/` rather than `tests/profiling/` because it
  imports `prolix` (which `tests/profiling/`'s seam discipline keeps it clear of) and because
  it pays a real XLA compile. It is marked `@pytest.mark.slow` so it is EXCLUDED from CI's
  `-m "not (slow or integration or dynamics)"` expression — the assertion is a P2 gate leg
  run deliberately, not a per-PR cost. Record in the step notes the exact command used
  (`uv run pytest tests/physics/test_scope_inventory.py -q -m slow`) and its output, since CI
  will not re-run it.

Modify:
- `src/prolix/physics/flash_explicit.py` — wrap `_total_energy_fn` (bonded block),
  `chunked_explicit_nonbonded_energy` (tile loop), `_chunked_lj_only_energy`, and the
  `jax.grad` call inside `flash_explicit_forces._grad_fn`. Suggested labels:
  `flash_bonded`, `flash_nonbonded_tiles`, `flash_lj_only`, `flash_grad`.
- `src/prolix/batched_energy.py` — wrap the term calls inside `single_padded_energy`:
  `_bond_energy_masked`, `_angle_energy_masked`, `_dihedral_energy_masked`,
  `_cmap_energy_masked`, `_lj_energy_masked`, `_coulomb_energy_masked`,
  `_pme_reciprocal_and_corrections`. Suggested labels: `dense_bonded_*`,
  `dense_lj_direct`, `dense_coulomb_direct`, `dense_pme_reciprocal`.

Match the existing convention already present in `pme.py` (`pme_charge_spread`,
`pme_fft_forward`, …) and `settle.py` (`settle_pos_eigh`, …).

**Gate:** locally, `uv run pytest tests/physics/test_flash_explicit.py
tests/physics/test_batched_energy.py -q` passes — a narrow selector over the touched
modules, because whole-suite JAX runs are denied on this box by the local-compute guard and
P2 is declared a FREE/local step. The full check is CI's, using CI's own marker expression:
`uv run pytest -m "not (slow or integration or dynamics)"` must be unchanged on the PR
(matching `.github/workflows/ci.yml`, **not** `-m "not slow"`, so "unchanged" is measured
against the baseline CI actually produces). And `tests/physics/test_scope_inventory.py`
asserts, against the **compiled** module
(`jax.jit(fn).lower(*args).compile().as_text()`), not the lowering, at the largest size P4
executes (N = 64) — every label above must appear. Rationale:
`scripts/experiments/profile_b1_water_trace.py` lines 31-44 documents that at
replicas=2/n_steps=5 every `pme_*` label vanishes from compiled HLO while `settle_*` labels
survive, so a pre-optimization lowering is not evidence of survival. If a
`dense_pme_reciprocal`-class label fails this assertion at N = 64, escalate the assertion to
the P7 probe size before declaring the label unavailable.
**Terminal state when ANY label is invisible at N = 64 — stated per label class, because the
gate is not uniform.** (a) A `dense_pme_reciprocal`-class label: P2 is **conditionally
passed with a recorded deferral** — this is the expected path, not an edge case, per the
cited `profile_b1_water_trace.py` note. Escalate the assertion to the P7 probe size; declare
the label unavailable (triggering the op-name fallback) only if P7's Stage-2 assertion also
fails to find it. (b) Any OTHER label (`flash_nonbonded_tiles`, `flash_lj_only`,
`flash_bonded`, `flash_grad`, `dense_lj_direct`, `dense_coulomb_direct`, `dense_bonded_*`):
P2 is **also conditionally passed**, but the deferral is recorded as `unexpected_absence`
rather than `deferred_to_p7`, and it carries one extra obligation — confirm from the
compiled text that the wrapped call site survived at all (i.e. distinguish "fused away and
the label with it" from "the wrapping never landed"), because the second is an
implementation error in P2 itself and IS a hard block. **In neither case does P2 block
P3–P6.** An absent label is already a defined downstream outcome: P4 records it as
`scopes[label] = None` and falls back to op-name attribution (see P4's fallback), and P7
does the same. This step's real gate remains P5.
**Scope: ~40 LOC of wrapping, no logic change.**

**Step notes (completed 2026-08-17):** All 11 wraps landed (7 dense in `batched_energy.py`'s
`single_padded_energy` -- bond/urey_bradley/angle/dihedral/improper/cmap split individually
rather than one `dense_bonded` blob, plus lj/coulomb/pme_reciprocal; 4 flash in
`flash_explicit.py` -- `flash_bonded` in `flash_explicit_total_energy`'s actual bonded-terms
block, not `_total_energy_fn` as the spec's Modify list named, since that function is
purely the nonbonded/electrostatics dispatcher; `flash_nonbonded_tiles`, `flash_lj_only`,
`flash_grad`). `tests/physics/test_flash_explicit.py` referenced by this section's Gate does
not exist in the repo -- substituted the actually-relevant `tests/physics/test_flash_dense_parity.py`
+ `tests/physics/test_batched_energy.py` as the narrow local gate (both pass, 5/5, unaffected
by pure scope-wrapping). Created `tests/physics/test_scope_inventory.py` (6 tests) using the
existing `_four_water_bundle()` fixture (`scripts/benchmarks/b1_init_exec.py`), which pads to
N=64 (ATOM_BUCKETS floor) -- exactly the size this gate calls for. Command:
`uv run pytest tests/physics/test_scope_inventory.py -q -m slow` → `6 passed`. Findings:
- All 5 dense labels backed by real (non-zero-mask) terms in this fixture (bond,
  urey_bradley, angle, lj_direct, coulomb_direct) and all 3 flash labels reachable at N=64
  survive in compiled HLO -- no unexpected absences, no wrapping-never-landed cases.
- `dense_pme_reciprocal` **does** survive in compiled HLO at N=64 for this fixture --
  better than this gate's own worst-case expectation (drawn from `profile_b1_water_trace.py`'s
  toy-scale finding); recorded, not asserted, per case (a).
- `dense_bonded_dihedral`/`dense_bonded_improper`/`dense_bonded_cmap` do not survive at
  N=64 for this specific fixture -- rigid TIP3P water has zero real (pre-padding) entries for
  these three terms, so their bucket-padded all-mask-zero contribution is constant-folded
  away by XLA with nothing left in HLO to label either way. Verified via source-level check
  (not HLO) that all three wraps genuinely exist in `batched_energy.py`, ruling out the
  "wrapping never landed" hard-block case. `flash_lj_only` required a dedicated EFA-method
  compile (unreachable via the default PME path both `flash_explicit_total_energy` and
  `flash_explicit_forces` use).
- CI-marker check: `pytest tests/physics/test_scope_inventory.py --collect-only -m "not (slow or integration or dynamics)"`
  → `no tests collected (6 deselected)`, confirming the module-level `@pytest.mark.slow` keeps
  this file out of CI's fast suite. The full-suite `unchanged` comparison is CI's own job
  (whole-suite runs are blocked locally by this box's compute-limits guard).

---

### P3 — Stage 0 probe: `cost_analysis()` dense vs flash, no execution
**Cost: FREE.** Structural claims only.

Create:
- `scripts/explore/prof_stage0_cost_analysis.py` — for tiny synthetic systems (N = 4, 16,
  64), lower and compile both force paths
  (`prolix.batched_energy.single_padded_energy` + `jax.grad`, and
  `prolix.physics.flash_explicit.flash_explicit_forces`) via
  `jax.jit(fn).lower(*args).compile().cost_analysis()`, and emit `ProbeRecord(stage=0, ...)`
  to `outputs/profiling/stage0/`. Stamp `platform="cpu"` on Stage-0 records — the lowering
  target, and the honest answer for a probe that does not execute. The choice is inert for
  every guard (a Stage-0 record carries neither `total_step_seconds` nor `scopes`, so
  `select_sources` filters it out of `TERM_RANKING` and `END_TO_END` before guard (d) is
  reached, and `STRUCTURAL`/`DISPATCH_COUNT` do not invoke unanimity); it is pinned anyway so
  two probes do not stamp it differently for no reason. Map `cost_analysis()` into `metrics`
  with a fixed key convention: take the returned mapping (JAX returns a dict, or a
  single-element list of dicts on some versions — normalise the list form to its sole element
  and raise if it has more), and stamp `flops`, `bytes_accessed`, and `optimal_seconds` under
  exactly those names, omitting any the installed JAX does not provide. Any further keys are
  stamped verbatim under their JAX-given names. Record in the step notes which keys the
  installed JAX actually returned, since these are version-fragile in the same way P4's
  dispatch counters are.

**Gate:** script runs to completion locally; emits ≥6 Stage-0 records; a smoke assertion
confirms `assert_claim_supported(records, ClaimClass.TERM_RANKING)` **raises** AND that the
raise names the metric/scopes filter, not the stage floor — Stage-0 records carry neither
`total_step_seconds` nor `scopes`, so the fail-closed selector fires first and the stage
floor is never reached. This gate demonstrates that the contract is live on real records; it
does NOT demonstrate the stage rule (P4's gate demonstrates that, on records that actually
reach guard (a)).
**Scope: ~120 LOC.**

**Step notes (completed 2026-08-17):** Synthetic systems are hand-built `PhysicsSystem`
instances at literal N=4/16/64 (not `MolecularBundle`/`ATOM_BUCKETS`-padded -- the bucket
floor is 64, which would make N=4 and N=16 indistinguishable from N=64 and defeat this
probe's whole point of comparing structural cost across genuinely different sizes), zero
bonds/angles/dihedrals/impropers (keeps every host-side topology-processing branch, e.g.
`topology.find_bonded_exclusions`, inactive via its `sys.bonds.shape[0] > 0` static check),
periodic with PME (`box_size=[50,50,50]`, `pme_alpha=0.3`). Emitted 6 records (dense × 3
sizes, flash × 3 sizes) to `outputs/profiling/stage0/`; smoke check passed (raised naming
"required metrics", not a stage-floor message).

**`cost_analysis()` keys on the installed JAX** (version-fragile, per this section's own
note): returns a plain `dict` (not a list) with `"flops"` and `"bytes accessed"` (space, not
underscore -- mapped to `bytes_accessed`) present; **no `"optimal_seconds"` key at all** on
this install (omitted, per the spec's own "omitting any the installed JAX does not
provide" rule). Additional keys stamped verbatim: `"transcendentals"`, plus a large
per-operand/per-output breakdown (`"bytes accessed{N}{}"`, `"bytes accessedout{}"`,
`"utilization{N}{}"` for N up to the compiled program's per-op count -- 92 such keys at
N=64 flash, since flash's HLO has far more discrete ops than dense's).

**Structural finding (the actual free/no-execution payoff of this probe):** dense scales
with N as expected for an O(N²) kernel -- flops 6212 (N=4) → 98000 (N=16) → 1599296 (N=64),
roughly N² scaling. **Flash's flops stay essentially flat** across all three sizes
(~7.003e8 at N=4, N=16, AND N=64) -- because FlashMD's fixed tile size (T=256 default)
pads any N ≤ 256 up to exactly one full T×T tile (`n_tiles = ceil(N/T) = 1` for all three),
so the compiled graph's cost is dominated by the tile size, not the real atom count, at
this scale. This is a genuine structural property of the tiled architecture (not a probe
artifact), and load-bearing context for later stages: any FlashMD-vs-dense cost comparison
below N≈256 (P4/P6/P7 all measure well under that) will show flash's *fixed* per-tile cost
dominating, not its real per-atom work -- the crossover where flash's O(N²/T) advantage
actually materializes is structurally invisible until N exceeds one tile.

---

### P4 — Stage 1 probe: executed CPU micro-probe with trace capture
**Cost: CHEAP** (N = 4–64, seconds; local box constraint respected).

Create:
- `scripts/explore/prof_stage1_cpu_micro.py` — executes a handful of integrator steps at
  N = 4–64 under `jax.profiler.trace`, parses the trace into per-`named_scope` wall times,
  and emits `ProbeRecord(stage=1, platform="cpu", ...)`.
- `scripts/profiling/trace.py` — **pure** trace→`dict[scope, tuple[seconds, n_occurrences]]`
  parsing (no prolix imports); the return type is the tuple form throughout, matching
  `ProbeRecord.scopes` exactly. There is no seconds-only variant. **Input artifact:** the
  `*.trace.json.gz` Perfetto JSON emitted into the `jax.profiler.trace` output directory —
  chosen over `.xplane.pb` because it needs no TensorBoard plugin dependency, which would
  violate the xtrax-shaped seam. **Convention:** returns **exclusive** (self) time per scope
  — a nested child's time is subtracted from its parent — and **sums** across re-entrant
  occurrences (every `scan`/`while_loop` iteration of the same label aggregates into one
  entry). Inclusive time is derivable from the parent/child tree and is not the default,
  because a bottleneck table built from inclusive times double-counts.
  **Input structure (resolve before writing the parser, not after):** dump one real
  `jax.profiler.trace` output first (`profile_b1_water_trace.py --trace` already produces
  one) and record in the step notes HOW a `named_scope` label surfaces — as a nested
  duration slice with a real parent/child stack, as a `/`-delimited prefix on an XLA op
  name, or on a `TraceAnnotation`/`XLA Modules` row. The exclusive-time convention above is
  only defined for the nested-slice form; if labels surface as flat name prefixes instead,
  state that exclusive time is computed by prefix-depth attribution instead, and say so in
  the record.

Reuse `scripts/experiments/profile_b1_water_trace.py` as the working reference for
`jax.profiler.trace` capture mechanics — do not re-derive it.

**Gate:** produces a per-scope breakdown covering the labels added in P2, plus
**dispatch counts**, defined operationally as four separately-reported integers, because
they are four different diagnostics: (1) `n_executions` — XLA executable invocations over
the timed region; (2) `n_compilations` — distinct XLA compilations triggered, the
retracing/recompilation tripwire; (3) `n_host_syncs` — `block_until_ready`/device→host
transfers; (4) `n_jit_traces` — Python-level trace events. All four are stamped into
`metrics`. **Mechanism, per counter, all derived from the same Perfetto trace this step
already captures so no second measurement path is introduced:** `n_executions` from
executable-invocation events; `n_compilations` from XLA compile events (cross-check against
`jax.monitoring` compilation-event listener counts where available); `n_host_syncs` from
device-to-host transfer events; `n_jit_traces` from jit trace-annotation events. All four
are trace-event-derived and therefore version-fragile — assert on their PRESENCE, not their
exact spelling, and pin the JAX/jaxlib version in the record (already stamped).
**Resolve counter feasibility as a spike BEFORE P3 or P4 emits any record.** Against the
installed JAX, capture one throwaway `jax.profiler.trace` of a trivial jitted function and
confirm each of the four counters is derivable from it. This is minutes of work and it must
precede the first emitted record, because the fallback below takes a MAJOR
`CONTRACT_VERSION` bump, and read-side rule (iii) makes records at different MAJOR versions
unmixable — so a fallback that fires *after* P3's Stage-0 records exist silently strands
them from every later record, which is precisely the cross-stage load P8 performs. Record
the spike's result in the step notes. **If the fallback nevertheless fires after any record
has been written**, that is a bump-invalidation event under P1's bump rule: every record at
the prior MAJOR version is retained as history but is not citable and is not mixed into a
source set; re-emit the Stage-0/1 records (minutes) and regenerate P8's fixtures at the new
version.
**Fallback, decided now rather than at gate time:** if a counter proves unobtainable on the
installed JAX, remove it from `REQUIRED_METRICS[DISPATCH_COUNT]`, record the removal and its
reason in the step notes, and take a MAJOR `CONTRACT_VERSION` bump — removing a required
metric newly ACCEPTS previously-rejected claims, which is a major-bump trigger under P1's
rule. Do not ship a `DISPATCH_COUNT` frozenset containing a metric no probe can emit;
`select_sources` is fail-closed and would raise on every such claim forever. A
`DISPATCH_COUNT` claim must name which of the four it ranges over. Records are
Stage-1 stamped and consequently *cannot* be used for PME-vs-nonbonded ranking. Claims
permitted here: structural, dispatch-count, and bonded/SETTLE relative cost only.
And `assert_claim_supported(stage1_records, ClaimClass.TERM_RANKING)` raises with the
STAGE-FLOOR message — P4's records carry `total_step_seconds` and ≥2 non-`None` `scopes`,
so they are the first real records that reach guard (a). This is where the stage rule is
demonstrated on real records (contrast P3's gate, which cannot reach guard (a) at all, since
Stage-0 records carry neither metric).
**Fallback:** any P2 label absent from the executed trace is recorded as `scopes[label] =
None` (not 0.0), and attribution for that term falls back to trace-level op-name
aggregation — the same fallback P5 specifies — written into `scopes` with
`attribution_method[label] = "op_name"`; the `None` sentinel is reserved for a label
recovered by NEITHER method. A missing label is a recorded outcome, never a silent zero.
**Parser ground-truth check:** `tests/profiling/test_trace_parse.py` has TWO legs.
(1) synthetic: a hand-constructed trace with known nesting and known durations (one parent,
two children, one child entered 3×), asserting the parser recovers the exact exclusive
seconds and occurrence counts. (2) real: a small (<1 MB) `*.trace.json.gz` committed under
`tests/profiling/fixtures/`, produced **locally on CPU** by
`JAX_PLATFORMS=cpu uv run python scripts/experiments/profile_b1_water_trace.py --trace
--replicas 2 --n-steps 5 --n-trials 2` (the reduced parameters are what keep it under 1 MB;
the script defaults to `--replicas 16 --n-steps 50 --n-trials 5`, which does not). Record the
exact command in the fixture directory's README. Over it, the parser must recover at least
two of `pme.py`/`settle.py`'s pre-existing labels with non-zero exclusive time. **Stated
limitation, because it bounds what this leg proves:** the fixture is a CPU trace, so it
validates the parser against the CPU artifact shape only. `--trace` is documented as GPU
device trace capture and the two artifact shapes may differ (different device-lane rows,
different XLA op-name decoration). A GPU fixture is not obtainable inside P4, which is a
local CPU step. Therefore **P7 carries an additional gate leg**: run `trace.py` over one real
GPU trace captured during the sweep and assert it recovers ≥2 labels with non-zero exclusive
time, before any P7 per-scope attribution is cited. If the GPU shape differs, that is a
parser defect discovered at P7 and it invalidates P7's per-scope attribution only — P7's
dense-vs-flash term ranking survives, on the same reasoning the scopes-equivalence failure
branch already gives. Leg (1) alone cannot detect a parser written against an assumed
artifact shape. Per the project rule on verifying a measurement pipeline before
trusting conclusions, this test gates `trace.py`, not the probe output. **Fallback:** if the
CPU trace lacks per-scope granularity for a given label, that label is recorded as `None`
and falls back to op-name attribution (see the label-absence fallback above). The gate list
also includes: a record with one absent label round-trips through JSON with that label
still `None`, asserted with `is None`, not falsy.
**Scope: ~180 LOC.**

**Step notes (completed 2026-08-17):**

*Pre-record feasibility spike (required before P3 or P4 emits any record):* captured a
throwaway `jax.profiler.trace` of a trivial jitted function on this JAX install and confirmed
which of the four dispatch counters are derivable. `n_executions`
(`CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice` events -- NOT
`ExecuteReplicated.__call__`, which does not scale 1:1 with actual dispatch count on this CPU
backend, confirmed directly: 5 calls to the same jitted function left it at 1 while
`ExecuteHelperOnSingleDevice` correctly tracked all 5), `n_compilations`
(`backend_compile_and_load` events, cross-checkable against the official
`/jax/core/compile/backend_compile_duration` `jax.monitoring` event), and `n_jit_traces`
(`PjitFunction(<fn_name>)` events, cross-checkable against `/jax/core/compile/jaxpr_trace_duration`)
are all cleanly derivable. **`n_host_syncs` is not** -- confirmed by direct experiment: calling
`.block_until_ready()` 7 times on an already-computed result produces *zero* additional trace
signal (no event count scales with the repeated calls), because CPU device buffers ARE host
memory, so there is structurally nothing for a "device->host transfer" event to represent on
this platform. Per the fallback this section specifies, `n_host_syncs` was removed from
`REQUIRED_METRICS[ClaimClass.DISPATCH_COUNT]` in `claims.py`, and `CONTRACT_VERSION` was bumped
`2.0` -> `3.0` (removing a required metric newly *accepts* previously-rejected claims, the
spec's own MAJOR-bump trigger). Since this fallback fired *after* P3's Stage-0 records already
existed (a fully honest miss of this section's own "resolve BEFORE P3 or P4 emits any record"
instruction -- the spike was run at the start of P4's work, not before P3's), P3's 6 records
were re-emitted at the new version per the bump-invalidation remediation this section specifies.

*How a `named_scope` label actually surfaces (resolved before writing the parser, per this
section's requirement):* dumped a real trace via a toy `jax.named_scope`-wrapped function, then
a real multi-step water-plan program (`profile_b1_water_trace.py`'s own builders). Neither the
nested-duration-slice form nor a `TraceAnnotation`/`XLA Modules` row applies. The **executed**
Perfetto trace's per-thunk events are named after the *post-fusion* HLO thunk/op name (e.g.
`"wrapped_multiply"`, `"reduce_add_fusion"`), carried in `args["hlo_op"]` -- these names alone
carry zero `named_scope` information. The scope path survives only in the *separately-obtained*
compiled HLO text, as a `/`-delimited prefix on each instruction's
`metadata={op_name="jit(fn)/scope1/scope2/.../leaf_op"}` field. `trace.py`'s parser is therefore
two-input, not trace-only: `scope_map_from_hlo_text(hlo_text, known_labels)` builds an
instruction-name -> scope-path map by resolving each instruction's own metadata, or (when absent,
e.g. a top-level `while`-body/fusion-calling instruction) following `calls=`/`to_apply=`
references down to a callee computation's `ROOT` instruction, recursively; `parse_scopes(trace_events,
scope_map)` then looks up each trace event's `hlo_op` in that map. This is exactly the spec's own
anticipated fallback ("if labels surface as flat name prefixes instead, state that exclusive
time is computed by prefix-depth attribution instead") -- confirmed as the real shape, not the
nested-slice default the synthetic-leg description names generically. One further wrinkle found
only on the real (vmapped + autodiffed) program, not the toy one: under `vmap`/`jvp`, a scope
path segment is decorated as `"vmap(jvp(dense_bonded_improper))"`, not the bare label --
`_unwrap_transform` peels these `word(...)` wrappers recursively before matching against
`known_labels`. Recorded as a durable lesson (praxia lesson id 366) for P9's skill authoring.

*Real-fixture leg finding, and the resulting scope-of-gate decision:* the committed fixture
(`tests/profiling/fixtures/b1_water_stage1.trace.json.gz`, `--replicas 2 --n-steps 5 --n-trials 2`,
646 KB) recovers **P2's own `dense_bonded_*`/`dense_lj_direct`/`dense_coulomb_direct` labels**
with non-zero exclusive time (7 labels) -- but recovers **none** of `pme.py`'s/`settle.py`'s
pre-existing labels, which is what this section's Gate literally names. This is a *distinct*
phenomenon from the already-documented "vanishes from HLO text entirely at toy scale" `pme_*`
limitation: `settle_pos_eigh`/`settle_cramers_rule` *do* survive in the compiled HLO text (46/309
occurrences at this scale, confirmed via `--hlo-only`'s own analysis), but their instructions get
fused into larger blocks whose top-level (execution-visible) thunk carries some *other*
instruction's representative `op_name` metadata -- so they never surface as an attributable
top-level thunk in the *executed* trace, even though present in the HLO. Neither increasing
`--n-steps` (fusion granularity is a compiler decision independent of loop trip count) nor a
larger `--replicas` (blows past the 1 MB fixture budget) resolved this. **User decision
(2026-08-17):** accept `dense_bonded_*` recovery as satisfying this leg's real purpose (proving
the two-input trace+HLO attribution pipeline recovers real, non-zero wall-clock time on a real
program -- exactly what a synthetic-only leg 1 cannot demonstrate) for now; filed as backlog
#4330 (depends on #4322/P4) to investigate whether `settle.py`/`pme.py`'s `named_scope` placement
needs to change to survive execution-thunk-level attribution too, needed before any settle/pme
term-level wall-clock claim (P5/P7/P8) could be trusted the same way `dense_bonded_*` claims now
can be.

*Gate confirmed on real records (the whole point of P4's gate, contrasting P3's):*
`scripts/explore/prof_stage1_cpu_micro.py` emits a real Stage-1 record (`total_step_seconds`,
7 non-`None` scopes, `n_executions`/`n_compilations`/`n_jit_traces` all present) and
`assert_claim_supported([record], ClaimClass.TERM_RANKING)` raises naming the **stage>=2 floor**
(not the metric/scopes filter, unlike P3) -- confirmed by direct run.

---

### P5 — Perturbation check: do the new scopes change fused hot-path timing?
**Cost: CHEAP.** This is the *CPU* gate on P2. It is necessary, not sufficient: fusion
decisions are backend- and shape-dependent, so a clean CPU/N=64 result does not transfer to
H200 at Stage-2 size — by the same argument §1 uses to invalidate cross-platform term
ranking. P7 carries the Stage-2 replication.

Create:
- `scripts/explore/prof_stage1_scope_perturbation.py` — A/B total step time at N = 64, ≥5
  repeats. The scopes-off arm is produced **inside this script**, by patching
  `jax.named_scope` to a null contextmanager before the prolix modules under test are
  imported (`contextlib.nullcontext`-equivalent). No env var and no source-side toggle: the
  A/B is a property of the harness, so P2 keeps its `no logic change` guarantee and prolix
  gains no permanent hot-path config surface with no owner. The **assertion** is
  `numpy.allclose(F_on, F_off, rtol=<r>, atol=<a>)` computed elementwise over the full (N,3)
  force arrays — not a threshold on any scalar summary. `(r, a) = (1e-5, 1e-6)` when the
  children ran under float32 and `(1e-12, 1e-14)` under x64; the branch is selected from the
  child records' auto-captured `x64_enabled`, which the driver asserts is identical across
  both arms before comparing (a cross-precision A/B is void regardless of the delta).
  `max_abs_force_delta = float(np.max(np.abs(F_on - F_off)))` is computed and stamped into
  the paired record's `metrics["ab_max_abs_force_delta"]` as **reportage only** — it is what
  makes a nonzero-but-in-tolerance fusion change visible and comparable across runs, and it
  is deliberately not the pass/fail quantity, because a relative tolerance has no meaning
  against an unreferenced absolute maximum. Deliberately not bitwise: a no-op scope cannot
  change semantics, but it can change XLA fusion, and a changed fusion changes reduction and
  FMA-contraction order, so last-bit differences are the expected signature of exactly the
  perturbation this gate exists to detect. `allclose` False means the patch is wrong (it
  disabled or reordered real work) and the timing comparison is void; `allclose` True with a
  nonzero `max_abs_force_delta` is itself reportable evidence that fusion changed, and is
  noted alongside the timing verdict.

*Process model (specified, because an import-time patch cannot be flipped in-process).* Each
arm runs in its own subprocess. `prof_stage1_scope_perturbation.py` is a driver that
alternately spawns `--arm scopes_on` and `--arm scopes_off` child invocations of itself,
A/B/A/B. The child applies (or omits) the patch before importing prolix, performs 3 untimed
warm-up calls that absorb JIT compile entirely inside the child, then `n_inner` timed
repeats, and reports the median of its timed repeats. Compile cost therefore never enters a
reported number, and the two arms' differing HLO compile times cannot enter `d_i`. Pair *i*
is the *i*-th A child with the *i*-th B child. Each child takes `--system-seed` and
constructs its N=64 system from that seed alone, so both arms are bit-identical inputs; the
driver passes the same value to both children of a pair and stamps it into
`config["system_seed"]`. Each child writes `forces.npy` (float64, shape `(N,3)`) beside its
`ProbeRecord` at `<record_path>.forces.npy`. The driver loads both arms' arrays, computes
`max_abs_force_delta`, asserts it against the stated tolerance, and stamps it into the PAIRED
summary record's `metrics["ab_max_abs_force_delta"]` — the children cannot compute it because
neither sees the other. `n_inner = 20` timed repeats per child (after the 3 untimed
warm-ups), stamped into `metrics["n_inner"]` alongside seed/`B`/`n_pairs`; `method`
("percentile bootstrap") is stamped into `config["method"]`, not `metrics` — it is a
string, and `ProbeRecord.metrics` is float-only by contract (P1). Each child
writes its own `ProbeRecord`; the driver emits the paired summary.

**Gate:** paired equivalence on total step time **and** on the instrumented function in
isolation. Procedure: ≥21 interleaved A/B pairs (see the process model above), the first 2
pairs discarded as page-cache/thermal warm-up, leaving ≥19 paired deltas
`d_i = (t_scopes_i − t_noscope_i) / t_noscope_i`. Pass requires `|median(d_i)| ≤ 0.05`
**and** the 90% percentile-bootstrap CI of `median(d_i)` contained in [−0.05, +0.05],
computed with `B = 10,000` resamples using `numpy.random.default_rng(20260817)`. The seed
and `B` are stamped into the emitted record's `metrics`; `method` is stamped into `config`
(see the process-model paragraph above for why it cannot live in `metrics`); together with
`n_pairs` (also in `metrics`), the verdict is reproducible from the record alone; a gate
that flips on an unseeded resample is not a gate.
Run the same check a second time on
`chunked_explicit_nonbonded_energy` timed alone, because a perturbation confined to the tile
reduction can hide under 5% of a whole step — which is the exact mechanism this gate exists
to catch. Both checks must pass. **Distinguish the two failure modes before acting.** If
`|median(d_i)| > 0.05`, that is a DETECTED perturbation — revert the offending scopes and
fall back to op-name attribution. If `|median(d_i)| <= 0.05` but the CI is not contained in
the margin, that is INSUFFICIENT POWER — increase `n_pairs` and re-run; DO NOT revert. **A
DETECTED perturbation is localised, not blanket-reverted — see the localisation procedure
after this paragraph.** Size
`n_pairs` from a pilot: run 10 A/A pairs first (both arms scopes-off; 10 rather than 5
because an IQR estimated from n=5 is too unstable to size anything), take `s` = the
interquartile range of those null deltas, and set
`n_pairs = clip(ceil((3.3*s/0.05)^2), 21, 200)`. The constant 3.3 is a deliberately
conservative stand-in for the ~1.5 a normal-theory derivation gives (sigma ≈ IQR/1.349; 90%
CI half-width on a median costs a further sqrt(pi/2)) — roughly 2× on the coefficient and ~5×
in `n`, erring toward more power because the insufficient-power branch forbids reverting and
a re-run is cheaper than a wrong revert. Stamp the pilot `s`, the raw derived value, and the
clipped `n_pairs` into the record. **When the raw value exceeds the 200 cap** the pilot is
telling you the harness noise floor is comparable to the effect the gate is meant to resolve,
and no affordable number of pairs will settle it. Do not revert (that branch is forbidden),
and do not raise the cap. Instead: record the outcome as UNRESOLVED-ON-CPU with the pilot
`s`, run the gate at `n_pairs = 200` and report the resulting CI without a pass/fail verdict,
and defer the verdict to P7's Stage-2 replication — which is the backend and size where the
attribution is actually believed. P2 is then carried forward under the same
conditional-pass-with-recorded-deferral terms it already has for an invisible label. A gate
that deletes instrumentation because the measurement was noisy is
measuring the harness, not the code. Record the outcome in the spec's follow-up notes either
way — this is a real decision point, not a formality. Scopes can perturb fusion inside a
fused `scan`/tile-reduction, which is exactly what `chunked_explicit_nonbonded_energy` is.

**Localising a DETECTED perturbation.** The two checks already partition the space once: if
the `chunked_explicit_nonbonded_energy` check fails, the offender is that function's labels
(`flash_nonbonded_tiles`, and `flash_lj_only` where it shares the tile body) and those are
what get reverted. If the whole-step check fails while the per-function check passes, the
offender is elsewhere and the remedy is a **bisect over scope groups**, not a blanket
revert: the three groups are (i) `flash_explicit.py`'s bonded + grad labels
(`flash_bonded`, `flash_grad`), (ii) `flash_explicit.py`'s nonbonded labels
(`flash_nonbonded_tiles`, `flash_lj_only`), (iii) `batched_energy.py`'s dense term labels
(`dense_bonded_*`, `dense_lj_direct`, `dense_coulomb_direct`, `dense_pme_reciprocal`). The
harness patch is already all-or-nothing per import, so a group is disabled by reverting its
wrapping in a scratch branch and re-running the whole-step check; three re-runs at the
pilot-derived `n_pairs` bound the search. Revert only the group that reproduces the failure,
and fall back to op-name attribution for that group's terms alone. **If the bisect is
inconclusive** (no single group reproduces it), revert all P2 scopes and record that the
whole-step perturbation was not attributable — but state that outcome explicitly rather than
reaching it by default.
**Scope: ~90 LOC.**

**Step notes (completed 2026-08-17):** `scripts/explore/prof_stage1_scope_perturbation.py`
implements the process model exactly as specified (subprocess-per-arm, A/B/A/B, `--child`
mode patches `jax.named_scope` before importing prolix), with two real bugs found and fixed
during smoke-testing before any statistical run was trusted:

1. **The timed functions were not `jax.jit`-wrapped.** `eqx.filter_grad`/`jax.grad` alone do
   not compile; without an explicit `jax.jit`, every call (including the 3 "warm-up" ones)
   ran in eager per-op dispatch mode -- confirmed directly: median call time ~0.86 s on a
   trivial N=64 system, ~280x slower than the compiled path, and "warm-up absorbing JIT
   compile" absorbed nothing because there was no compile to absorb. Since fusion is
   fundamentally a compile-time decision, an un-jitted measurement cannot detect a fusion
   perturbation at all -- this would have silently produced a meaningless PASS on every run
   regardless of what P2 actually did. Fixed by wrapping the target function in `jax.jit`
   before the warm-up loop (median call time dropped to ~1-3 ms, consistent with a real
   compiled N=64 call).
2. **`jax.named_scope`'s scopes_off patch used `contextlib.nullcontext`, which is not
   callable as a decorator.** `equinox`'s own `eqx.nn` module decorates methods with
   `@named_scope(...)` at *import time* (not just context-manager usage) -- `nullcontext`
   instances support `with x:` but not `@x`, so any prolix import chain that pulls in
   `equinox.nn` crashed immediately under the scopes_off patch. Fixed with a small
   `contextlib.ContextDecorator` subclass (`_NullScope`) that supports both.

A third, environment-level (not code) bug surfaced while launching the actual gate runs:
background shell commands run under a **more restrictive filesystem sandbox** than
foreground ones on this box -- a path (`$CLAUDE_JOB_DIR/tmp`) freely writable all session in
foreground calls was read-only under `run_in_background`, silently killing two earlier
background attempts before any log line was written. A separate, related mistake (nesting
`nohup ... &` inside an already-backgrounded tool call) made the *tracked* command report
"completed" within ~1 s while the actual detached child was killed anyway -- `nohup`/`disown`
protect against SIGHUP, not whatever teardown mechanism ends the sandboxed session backing a
tool call. Fixed by running the real long-lived command as the sole foreground command of a
`run_in_background` call (no internal `&`), logging to a worktree-relative path instead of the
job tmp dir.

A fourth bug (found and fixed, not just worked around) was a genuine **race condition**:
running both targets as separate concurrent processes against one shared `out_dir` let them
both write `pair{i}_on.json`/`pair{i}_off.json`/`.forces.npy` at identical paths (pair indices
are process-local counters starting at 0 for every target), corrupting whichever pair happened
to collide. Fixed with a per-target subdirectory (`out_dir / target`) for all pilot/pair/summary
files -- required, not cosmetic, once two targets (or two re-runs) can run concurrently.

**Final gate results, both targets independently PASS:**

| target | verdict | median(dᵢ) | 90% CI | n_pairs | pilot s | `ab_max_abs_force_delta` |
|---|---|---|---|---|---|---|
| `whole_step` | **PASS** | −0.0040 | [−0.0215, +0.0394] | 50 (n_pairs override, see below) | 0.1160 | 0.0 |
| `tile_function` | **PASS** | −0.0111 | [−0.0202, +0.0026] | 59 (pilot-derived) | 0.1159 | 0.0 |

Both checks' 90% CI is fully contained in [−0.05, +0.05]; `ab_max_abs_force_delta = 0.0` on
every pair in both runs (forces bit-identical between arms, not merely within tolerance) --
the strongest possible negative result for a fusion perturbation. **`whole_step`'s first
attempt at the pilot-derived floor (`n_pairs=21`) landed `INSUFFICIENT_POWER`**
(median −0.0037 -- comfortably inside the margin -- but CI [−0.0565, +0.0411] just missed
full containment on the lower bound). Per this section's own rule ("increase n_pairs and
re-run; DO NOT revert"), added a `--n-pairs` override flag to the script and re-ran at
`n_pairs=50`, which resolved cleanly to PASS -- confirming the first result was a genuine
power problem (this box's timing noise at the 10-pilot-pair sample size), not evidence of a
perturbation. (A still-earlier `whole_step` attempt, run concurrently with `tile_function`
before the per-target-subdirectory race-condition fix above, produced a plausible-looking but
untrustworthy `INSUFFICIENT_POWER` summary and was discarded rather than reported.)

**Conclusion: P2's `named_scope` wraps do not perturb fused hot-path timing on CPU at N=64,
on either check.** No localisation/bisection was needed (no DETECTED_PERTURBATION verdict on
either target). P7's Stage-2/H200 replication (this section's own caveat: "a clean CPU/N=64
result does not transfer to H200 at Stage-2 size") remains the governing check for GPU.

---

### P6 — Re-verify the Stage-2 probe script runs end-to-end, and record its true `n_atoms`
**Cost: FREE→CHEAP** (CPU, reduced size).

**Precondition (verify before P6's first `bth run`, not after).** The plan's entire
separation-of-history mechanism assumes `bth run --campaign <new_id>` overrides a sidecar's
hardcoded `[metadata].campaign_id` (`profile_b1_flash_vs_autodiff_forces.bth.toml` pins
`32d6574e`). Verify it: `bth campaign create` the new campaign, run one throwaway
`bth run --campaign <new_id>` of the script's `--dry-run` path, then
`bth sql "SELECT id, campaign_id, tags FROM read_parquet('~/.bth/catalog/runs/prolix/run_*.parquet') ORDER BY timestamp DESC LIMIT 1"`
and read `campaign_id` directly. **If the CLI flag wins:** proceed as written. **If the
sidecar's metadata wins, or the gate rejects the mismatch:** the no-sidecar-edit route is
unavailable, and the fallback is to run P6's smoke and P7's tasks 0–11 from a
`scripts/explore/` copy of the script under `--no-sidecar` (project hard rule 6 sanctions
`--no-sidecar` for that directory), with the ProbeRecord tree carrying all provenance and the
campaign association recorded in the record's `config`. Record the observed precedence in the
step notes either way — a later reader must not have to re-derive it.

Modify:
- `scripts/experiments/profile_b1_flash_vs_autodiff_forces.py` — add `--stage`,
  `--emit-record`, a CPU-sized smoke mode that reports per-(protein × mode) wall time, and
  `ProbeRecord` emission.
- **Plan of record: no sidecar edit.**
  `scripts/experiments/profile_b1_flash_vs_autodiff_forces.bth.toml` has historical runs
  under campaign `32d6574e` and the project's pre-registration discipline is never to patch
  a running sidecar. P6 adds a CPU-sized smoke MODE to
  `profile_b1_flash_vs_autodiff_forces.py` that runs a real (shortened) timing loop with
  `--n-trials 3`, invoked under `bth run` against the script's existing frozen sidecar and
  the NEW campaign id (see P7). The `--dry-run` path is NOT an alternative route for the
  timing-loop leg of this gate: it returns before any timing (script lines 151-157) and
  therefore supplies `n_atoms` but no wall time. Use `--dry-run` only for the `n_atoms` half
  if the timing loop cannot be made to run, and record that P7's walltime derivation is then
  blocked. `--no-sidecar` is not used: the script lives in `scripts/experiments/` and
  project hard rule 6 reserves `--no-sidecar` for `scripts/explore/`.
- **What the frozen sidecar's gate verdict means for this run, stated so it is not later
  mistaken for evidence.** The frozen sidecar's three outcomes range over `flash_speedup`,
  pre-registered against a GPU tiling hypothesis. P6's smoke runs on CPU at `--n-trials 3`,
  so the gate WILL evaluate that condition and WILL record a pass/marginal/fail. That verdict
  is **not citable for debt 761 or for any hypothesis**: it is a smoke-sized CPU number
  scored against a GPU-anchored threshold, and it is retained only as the mechanical
  byproduct of running an existing script under its existing sidecar. Three things make this
  safe rather than merely disclaimed: (i) the run carries the NEW campaign id, so
  `bth campaign review 32d6574e` — the artifact debt 761 is decided from — never shows it;
  (ii) the run additionally carries `--tag p6-cpu-smoke --tag not-citable`, so it is
  filterable out of any `bth sql` query; (iii) the new campaign's description records that
  P6's run is a plumbing check and that the sweep's live pre-registration is
  `prof_stage2_coverage.bth.toml`. If, contrary to (i), `--campaign` does not override the
  sidecar's `[metadata].campaign_id` (see the precedence check this step's precondition
  adds), P6's smoke is instead moved to `scripts/explore/` as a temporary copy and run with
  `--no-sidecar`, which project hard rule 6 sanctions for that directory — and the
  sidecar-editing prohibition is preserved either way.

This script was blocked by debt 765, debts 832/835, and the `_host_box_size` tracer leak —
**all fixed this session (commit 3d42805)** — so it is expected to run for the first time.
Confirm rather than assume.

**Gate:** the CPU smoke is run TWICE, once per protein (`--protein 1vii` and
`--protein 2gb1`) — the script's `--protein` flag takes a single value with a single default
(L127) and each invocation loads and times exactly one bundle, so one run reports one
protein. Both invocations complete without exception AND run a real timing loop (the
`--dry-run` path alone does not satisfy this gate — see above), and each reports the
**actual production solvated** `n_atoms` for its protein — obtained from bundle
construction alone via the script's existing `--dry-run` path (lines 140-157 already emit
`n_real_atoms`/`n_padded_atoms` before any timing), so the smoke mode shortens the *timing
loop* and never shrinks the *system*. `ProbeRecord.n_atoms` is stamped with `n_real_atoms`.
Each invocation emits its own `ProbeRecord`, so P6 produces two records and two bathos runs
(both under the new campaign id; see the sidecar-verdict note above). **`t_cpu` for P7's
walltime derivation is the MAXIMUM over both proteins and both modes** of the reported
per-(protein × mode) wall time — the slowest cell governs, since one `--time` covers the
whole array.
Then, rather than re-deriving the ratio by hand, **call the contract**:
`assert_claim_supported(p6_records, ClaimClass.END_TO_END, target_n_atoms=23558)` over the
records both proteins emitted, and catch `ClaimValidityError`. A raise is the >10 branch; a
clean return is the ≤8-or-near-miss branch, disambiguated by reading
`23558 / min(r.n_atoms)` off the records. This keeps the human procedure and
`SCALE_EXTRAPOLATION_LIMIT` in sync by construction — a later change to the limit or to the
min-reducer then changes P6's verdict automatically instead of silently desynchronising from
a number typed into this paragraph. Record both per-protein ratios alongside the contract's
verdict so a later claim narrowed to one protein can be re-evaluated on its own ratio without
re-running P6. **This is also the first time guards (b) and (c) fire on real (non-fixture)
records** — note the outcome in the step notes for that reason. The governing ratio remains
the LARGER of the two per-protein ratios (i.e. the smaller `n_atoms`), which is not a choice
but a restatement of what the contract already enforces: guard (c) reduces with
`min(r.n_atoms for r in sources)`, so a claim backed by both proteins is governed by the
smaller system. Branch on the governing ratio:
- governing ratio ≤ 8 → Stage 2 → Stage 3 extrapolation is permitted by the contract; proceed.
- governing ratio in (8, 10] → **near miss**. The memory-regime question is answered
  analytically, not by running at the target — Stage-3 execution is out of scope for this
  step and for this plan. Compute, for each protein and for DHFR, the dense pair-storage
  estimate `M(N) = N_pad^2 * 4 bytes` (float32; use 8 for x64), where `N_pad` is the bucketed
  padded count the probe reports as `n_padded_atoms`, and for DHFR use the smallest bucket
  ≥ 23,558. Compare both against `D = 0.8 * <device memory of the P7 target>` — for the H200
  (node4009) take 140 GB, so `D = 112 GB`; state the device and the figure used in the record,
  because the answer is device-dependent by construction. **If `M(probe) < D` and
  `M(target) < D`, the memory regime is the same and the near miss passes.** Otherwise treat
  as ratio > 10 and file the mid-scale rung. Record `M(probe)`, `M(target)`, `D`, the device,
  and the verdict in `ProbeRecord.metrics` as `dense_pair_bytes_probe` /
  `dense_pair_bytes_target` / `device_memory_budget_bytes`. This is deliberately an estimate
  of the dominant term only (it ignores PME grid and neighbor-list storage, both of which grow
  more slowly) and it is deliberately conservative via the 0.8 factor; it is used because the
  guard it feeds is one-directional — it can only refuse a claim, never approve one. If the
  estimate lands within 20% of `D` in either direction, treat the answer as unknown and
  therefore as ratio > 10.
- governing ratio > 10 → **a mid-scale rung is mandatory**; file it as a blocking follow-up
  before any Stage-3 claim (this is the pre-mortem's story-2 failure mode firing early,
  which is the cheap place for it to fire). File this follow-up when EITHER protein's own
  ratio trips >10, because either one entering the source set tightens the guard for every
  claim built on that set. Record both per-protein ratios so a later claim narrowed to one
  protein alone can be re-evaluated on its own ratio without re-running P6.
**Scope: ~80 LOC, no sidecar edit.**

**Step notes (completed 2026-08-18):**

*Campaign-id precedence (precondition).* `bth campaign create "p6-stage2-cpu-smoke"` returned short id `d6f44eab` (full `d6f44eab-f538-4bf7-a84b-fbed3215908b`). A throwaway `bth run --campaign d6f44eab` of this script's `--dry-run` path recorded `campaign_id = d6f44eab-f538-4bf7-a84b-fbed3215908b` in the catalog, **not** the sidecar pin `32d6574e`. **CLI `--campaign` wins.** No sidecar edit; no `scripts/explore/` fallback. Sidecar was not modified.

*CPU smoke (real timing loop, not `--dry-run`).* Two cataloged runs under campaign `d6f44eab`, tags `p6-cpu-smoke` + `not-citable`; `32d6574e` did not gain them. Timed callables were already `jax.jit`-wrapped. Flash path ran (debt 765 no longer blocks). Bathos sidecar `flash_speedup` scores (≈2.61× / 2.63×) are **not citable** for debt 761 — CPU, `--n-trials 3`, GPU-anchored threshold. ProbeRecords at `outputs/profiling/p6/{1vii,2gb1}/record.json`, `stage=1`, `platform=cpu`, `git_sha=3f7ea36` (clean).

| protein | n_real_atoms | n_padded_atoms | energy_only_ms | autodiff_ms | flash_ms | total_step_seconds | 23558/n_real |
|---|---|---|---|---|---|---|---|
| 1vii | 1963 | 5000 | 124.86 | 557.16 | 213.79 | 0.557 | **12.00** |
| 2gb1 | 1987 | 5000 | 78.52 | 449.28 | 171.02 | 0.449 | **11.86** |

*Contract.* `assert_claim_supported(records, ClaimClass.END_TO_END, target_n_atoms=23558)` **raised** (guards (b) and (c) on real records): `END_TO_END extrapolation ratio 12.00x exceeds SCALE_EXTRAPOLATION_LIMIT=10.0x (min source n_atoms=1963)`. Both proteins independently trip >10. **Mid-scale rung is mandatory** before any Stage-3 claim; filed as backlog **#4350**. Do not raise `SCALE_EXTRAPOLATION_LIMIT`.

*P7 walltime input.* `t_cpu` = max over proteins and modes = **0.557 s** (1vii autodiff). P7 not started this step.

---

### P7 — FIRST AND ONLY GPU STEP: Stage 2 small-system probe on H200 / `pi_so3`
**Cost: GPU**, short walltime, small system. **Not DHFR.**

Create:
- `scripts/slurm/prof_stage2_small_system.slurm` — modelled on
  `scripts/slurm/fit_bonded_hp4.slurm`; sources `scripts/slurm/_bth_env.sh`; invokes
  `uv run bth run python scripts/experiments/profile_b1_flash_vs_autodiff_forces.py`
  with `--campaign`; requests `--partition=pi_so3,mit_normal_gpu`, `--gres=gpu:h200:1`,
  and `--time=00:15:00`. **Inherit from the template only:** the `SLURM_SUBMIT_DIR`
  anchoring block, the `source scripts/slurm/_bth_env.sh` line, the array-index→config
  decode pattern, and the `bth run` invocation shape. **Do not inherit**
  `#SBATCH --nodelist=node4007,node4008` (it pins to the Blackwell RTX Pro 6000 nodes and,
  per that file's own comment, deliberately excludes node4009 — the H200 — so inheriting it
  makes this job unschedulable against `--gres=gpu:h200:1`), nor `--gres=gpu:1`,
  `--time=2:00:00`, `--mem=16G`, or the `fit_bonded_hp4` output paths. P7's own directives:
  `--partition=pi_so3,mit_normal_gpu`, `--gres=gpu:h200:1`, `--time` per the derivation
  below, and `--array=0-11%<k>` (task 12 is submitted separately — see the walltime note
  below). **The throttle `%k` is a courtesy cap on concurrent GPU
  claims, NOT a placement control:** with `--partition=pi_so3,mit_normal_gpu` the scheduler
  may place tasks on any H200-equipped node in either partition, and no SLURM mechanism in
  this submission constrains that. `<k>` is sized by checking pi_so3 H200 availability
  before submitting (`ssh engaging "scontrol show node node4009"`, read `Gres=` and
  `AllocTRES`) so the cap is sensible, but it is recorded as a cap, not a guarantee that
  tasks share a node. The script also carries the standard hostname-keyed `XLA_FLAGS` guard
  block (per `~/.claude/rules/CLUSTER.md` §2, as 9 other slurm scripts in this repo do and
  the chosen template does not). The guard block is carried for consistency with the 9 other
  slurm scripts in this repo, but note precisely what does and does not make rule (d) pass
  on this sweep. It is **not** the guard — a hostname-keyed conditional is by construction
  node-dependent. It is the `--gres=gpu:h200:1` request: node4007/node4008 are the Blackwell
  RTX Pro 6000 nodes (see the do-not-inherit note above), so an H200 gres request structurally
  cannot land on either hostname the guard keys on. The guard is therefore inert on every task
  of this sweep, `XLA_FLAGS` is uniformly the empty string, and guard (d)'s `xla_flags`
  unanimity passes for that reason. **If the gres request is ever relaxed, this reasoning is
  void.** Accordingly, `prof_stage2_coverage.py` asserts that all 13 records carry an
  identical `xla_flags` string before computing any coverage or ranking, and reports the
  distinct values if not. A sweep that fires the guard on a proper subset of its tasks is
  uncitable as a whole and cannot be repaired by re-running the odd task out — the remedy is
  to re-run the entire sweep with the flag forced identically (set `XLA_FLAGS` unconditionally
  in the slurm script) and to record that the guard fired.
- `scripts/experiments/prof_stage2_scope_ab.py` — a thin harness wrapper carrying the same
  import-time patch P5 defines: it takes `--scopes {on,off}` plus the probe's own config
  flags, applies the `jax.named_scope`→`nullcontext` patch before importing any prolix
  module when `--scopes off`, then calls
  `profile_b1_flash_vs_autodiff_forces.main()` with the forwarded args. The toggle stays in
  the harness exactly as P5 requires, so P2 keeps its no-logic-change guarantee and prolix
  gains no permanent hot-path config surface. **Task 12 runs BOTH arms inside a single array
  task**, invoking this wrapper twice (`--scopes on`, then `--scopes off`, interleaved
  A/B/A/B exactly as P5's driver does) and emitting the paired summary itself — see the
  array-mechanism note below for why this is one task, not two. Tasks 0–11 invoke the probe
  directly.
- **Coverage is decided by a 14th, post-hoc AGGREGATION run**, because a bathos outcome
  condition is evaluated against a single run's flat result keys — the canonical sidecar
  (`fit_bonded_hp4.bth.toml`) routes cross-cell comparison to `bth campaign review` for
  exactly this reason; no run in this sweep computes or emits a sweep-level scalar itself.
  Create `scripts/experiments/prof_stage2_coverage.py`, which after the array completes
  reads TWO sources and joins them: (i) the 13 `ProbeRecord`s under
  `outputs/profiling/stage2/`, and (ii) the array tasks' exit codes from the bathos catalog,
  via `bth sql "SELECT id, status, exit_code, tags FROM
  read_parquet('~/.bth/catalog/runs/prolix/run_*.parquet') WHERE
  list_contains(tags, '<campaign-slug>')"` (exit code is captured by `bth run` per project
  hard rule 1; it is not derivable from the record set, and record presence alone is not a
  substitute because a task can write a record and then fail). If the catalog is
  unreachable, the script FAILS rather than falling back to record-presence — a coverage
  number computed from a weaker criterion than the one the outcome conditions were
  pre-registered against is not the metric. It then computes
  `n_configs_with_valid_dense_flash_pair` (0–6) plus per-cell booleans, and emits them as
  flat scalars. Its sidecar is `scripts/experiments/prof_stage2_coverage.bth.toml` —
  stem-matched per project hard rule 2 — carrying the outcomes below.
  `n_configs_with_valid_dense_flash_pair` is defined explicitly as: the number of
  *comparison cells* — the 6 distinct {protein} × {pme/grid} combinations, i.e. {1vii,
  2gb1} × {pme_off, pme_on@1.0Å, pme_on@1.2Å} — for which **both** the dense and the flash
  array task exited 0 and emitted a Stage-2 `ProbeRecord`. Maximum achievable value: 6.
  `[outcomes.pass] condition = "n_configs_with_valid_dense_flash_pair >= 5"`;
  `[outcomes.marginal] condition = "n_configs_with_valid_dense_flash_pair BETWEEN 2 AND 4"`;
  `[outcomes.fail]` is the residual with `is_residual = true`. Task 12 (the paired-scopes
  task) is excluded from this metric and is gated separately by the P5 equivalence
  criterion. Per-config `flash_speedup` values live in `[result_schema]` as recorded data,
  **not** as gate conditions: P7 is discovery, so the sidecar gates *coverage*, not *effect
  size*. Anchored effect-size thresholds arrive with the §4 promotion row, one experiment
  per surviving hypothesis. **Trigger for the aggregation run, stated so it cannot be
  forgotten:** it is submitted with the array, as a dependent job —
  `sbatch --dependency=afterany:<array_job_id> scripts/slurm/prof_stage2_coverage.slurm` —
  so it runs once the array reaches a terminal state whether or not every task succeeded
  (`afterany`, not `afterok`: partial coverage is exactly what this run exists to measure).
  Record the dependent job id alongside the array job id.
- **Per-task sidecar resolution, stated explicitly because P7 spans three different
  sidecars.** Tasks 0–11 invoke `profile_b1_flash_vs_autodiff_forces.py` and therefore
  resolve to its EXISTING frozen sidecar (`profile_b1_flash_vs_autodiff_forces.bth.toml`,
  see P6); they run under the NEW campaign id via `--campaign`, which is what separates
  them from campaign `32d6574e`'s historical runs. Because P7 changes that script's flag
  surface while P6 freezes its sidecar, the frozen sidecar's hypothesis and
  `[result_schema]` describe the pre-P7 shape and are preserved as the historical record,
  NOT as a live pre-registration for these 12 runs. **The new axes go in the ProbeRecord, not
  in the bathos result keys.** Tasks 0–11 emit exactly the eight keys the frozen
  `[result_schema]` declares, unchanged; `pme`, `grid_spacing` and `scopes` are carried in
  `ProbeRecord.config` per the config vocabulary below, never as bathos result keys. This is
  deliberate and it is what makes the no-sidecar-edit rule survivable: the sweep's cross-cell
  structure lives in the record tree, where `paired_configs` and `prof_stage2_coverage.py`
  read it, and the bathos run records serve only as run provenance (job id, git state, exit
  code, campaign association) — which is exactly the role the P7 gate assigns them
  (`bth campaign review <id>` shows the runs associated). The bathos records are consequently
  NOT the route for cross-cell comparison in this sweep; `paired_configs` over the 12 records
  is. If, contrary to this, the bath gate rejects a run for any schema reason at all, the
  remedy is to move the affected invocation to a `scripts/explore/` copy under `--no-sidecar`
  (sanctioned by project hard rule 6 for that directory) and record the move — never to edit
  the frozen sidecar. Task 12 carries its own new sidecar,
  `scripts/experiments/prof_stage2_scope_ab.bth.toml`, stem-matched to
  `prof_stage2_scope_ab.py`. The live pre-registration for the sweep as a whole is the new
  campaign plus `prof_stage2_coverage.bth.toml`; record this in the campaign description so
  the discipline is preserved in substance, not only in letter. State the campaign id
  explicitly once `bth campaign create` returns it.

**GPU-target note (recon inverted the prior assumption).** The dispatch brief anticipated
"not H200 first". Live queue data collected during recon says otherwise: for short (<15 min)
jobs, `pi_so3` H200 showed **6 pending vs L40S's 158** — roughly 26× *lower* contention, and
`pi_so3`'s `PriorityTier`/`AllowGroups` advantage applies. `titanix` is disqualified: no JAX
installed. So the least-contended target *is* H200 on `pi_so3`, and this step follows the
recon, not the prior assumption. Listing `pi_so3` first lets the scheduler try the
high-priority option before falling back.

**Design requirement — hypothesis-sweep parameterisation.** One submission sweeps the full
cross product, so a partition-contention inversion costs one contention decision rather than
N.

*Config list (exact: 12 array tasks + 1 separately-submitted paired-scopes job):* `{1vii, 2gb1} ×
{dense, flash} × {pme_on, pme_off} × {grid spacing 1.0 Å, 1.2 Å}`. PME-off runs use
direct-space Coulomb only and exist to isolate the reciprocal term; grid spacing is swept
only on PME-on runs, so the true count is `2 × 2 × (1 + 2) = 12`. Add a fourth axis:
`× {scopes_on, scopes_off}` on **one** representative config (1VII, flash, PME-on, 1.0 Å) —
a Stage-2 replication of P5 on the backend and at the size where the attribution is actually
believed. This is 1 extra array task, not a doubling, and not a duplicated base config:
task 12 measures BOTH arms of the (1VII, flash, PME-on, 1.0 Å) cell itself (see the
`scope_ab.py` bullet above), rather than duplicating the base task and hoping the two land
under comparable conditions — running the arms as separate array tasks would not deliver
that property, because SLURM gives no guarantee that two array tasks share a node, a GPU,
or a load level (see the throttle note above). The base task (within 0–11) remains in the
12 and in the coverage metric; task 12 is excluded from it.

*Mechanism:* a **SLURM job array**, `--array=0-11` (12 base configs), one config per task,
index-to-config mapping in the slurm script. Not an in-job loop: an array isolates
per-config failure, and every task carries its own `bth run`, so a task that OOMs orphans
nothing. Task 12 (the paired-scopes job) is a second, separately-submitted `sbatch` — see the
walltime note below for why it cannot share the array's `--time`.

*Walltime:* `--time = round_up_to_5min(3 * t_cpu * n_trials_gpu / n_trials_smoke) + 10 min
compile allowance`, floored at `00:15:00`, where `t_cpu` is the slowest per-(protein × mode)
wall time P6 reports at the default PME-on / 1.0 Å configuration, `n_trials_smoke` is the
`--n-trials` value P6's CPU smoke runs with (set it to 3), and `n_trials_gpu` is the
`--n-trials` value P7 passes (set it to 20). The 3× factor and the 10 min allowance are
stated safety margins, not derived bounds: 3× covers GPU-vs-CPU per-step variance and node
contention, and 10 min bounds observed XLA compile times for this program class; record both
as assumptions in the slurm script comment. The smoke route MUST run a timing loop — see
P6's gate; `--dry-run` alone cannot supply `t_cpu`. The PME-off and 1.2 Å axes are
deliberately not timed at P6 and need no separate measurement: PME-off strictly removes the
reciprocal term and 1.2 Å strictly shrinks the grid, so both are bounded above by the timed
PME-on/1.0 Å config. Record the derived number and its inputs in a comment at the top of the
slurm script; do not carry the placeholder `00:15:00` into the submitted script without that
derivation.

**Task 12 is submitted separately and is not covered by the formula above.** The array
`--array=0-11%<k>` carries the derived `--time`; task 12 is a second `sbatch` of
`scripts/slurm/prof_stage2_scope_ab.slurm` with its own
`--time = round_up_to_5min(2 * n_pairs * (t_gpu_child + t_compile_child)) + 15 min`, where
`t_gpu_child` is one child's (3 warm-ups + n_inner timed repeats) wall time and
`t_compile_child` is the per-child XLA compile time, both read from the first P7 base task's
log before task 12 is submitted. If the resulting `--time` exceeds the 12h partition limit,
reduce `n_pairs` to the largest value that fits and record the reduction plus the resulting
CI width as an INSUFFICIENT POWER outcome under P5's two-failure-mode rule — never as a
DETECTED perturbation, and never by reverting scopes.

**Config vocabulary (normative, exact spellings — every array task stamps all five keys,
never omitted, never defaulted):** `protein` ∈ {`1vii`, `2gb1`}; `mode` ∈ {`dense`,
`flash`}; `pme` ∈ {`on`, `off`}; `grid_spacing` ∈ {`1.0`, `1.2`} when `pme == on` and the
literal string `"na"` when `pme == off` (never omitted, never `""`, so PME-off cells group
cleanly and never merge with a PME-on cell); `scopes` ∈ {`on`, `off`}. A task that cannot
stamp all five does not emit a record. `paired_configs` (P1) raises `ClaimValidityError` if
any selected source is missing the `axis` key or any `hold_fixed` key, naming the record and
the key — defaulting a missing key to `""` would merge distinct cells and report a
confounded pair as a pass.

*Partial failure:* each array task emits its own `ProbeRecord`. Records from tasks that
exited 0 are valid and citable **provided**
`paired_configs(records, ClaimClass.TERM_RANKING, axis="mode",
hold_fixed=("protein","pme","grid_spacing"))` is non-empty — a term ranking requires its own
comparison arm, not the whole sweep. The gate below means full coverage, not merely exit 0.

Modify:
- `scripts/experiments/profile_b1_flash_vs_autodiff_forces.py` — add `--mode {dense,flash}`,
  `--pme {on,off}`, `--grid-spacing`, and per-config timing output. **This is the step that
  adds the sweep flags**; P6 adds only `--stage`/`--emit-record`/smoke. Also re-signature
  `main()` as `main(argv: list[str] | None = None) -> int`, passing `argv` through to
  `parser.parse_args(argv)` (currently `def main() -> int` parses `sys.argv` internally at
  L132, so the P7 wrapper cannot forward anything). `main()` with no argument keeps its
  present behaviour, so the existing `if __name__ == "__main__"` entry point and every
  historical invocation are unchanged.

**Gate:** all 12 array tasks + task 12 (submitted separately) complete (exit 0) **or** the
surviving subset covers at least one full dense/flash arm pair, with the shortfall recorded;
emits `ProbeRecord(stage=2, platform="gpu", device_kind="h200")` records; `bth campaign
review <id>` shows the runs associated (not orphaned); and
`assert_claim_supported(records, ClaimClass.TERM_RANKING)` now **passes** — the first
trustworthy dense-vs-flash and PME-vs-nonbonded ranking in the project. And the Stage-2
scopes-on/scopes-off pair (task 12) satisfies P5's equivalence criterion at the P7 size.
**If it fails**, the P7 term ranking is still valid (both arms of every dense-vs-flash
comparison carry the same instrumentation) but per-scope attribution from P7 is not — record
that and fall back to trace-level op-name attribution, exactly as P5 specifies. This does
NOT bypass `select_sources`'s ≥2-non-`None`-`scopes` filter, which the pre-existing
`pme.py`/`settle.py` labels satisfy independently of P2. **Additionally (A17's parser
ground-truth gate leg):** `trace.py`, run over one real GPU trace captured during this
sweep, recovers ≥2 labels with non-zero exclusive time before any P7 per-scope attribution
is cited — see P4's parser ground-truth check.

**Two gates, two different questions, stated so they are not confused.** The STEP gate
above governs whether P7 is complete as an implementation unit — one full dense/flash pair
plus a recorded shortfall suffices, because the machinery is then demonstrated end-to-end on
real Stage-2 records. The SIDECAR gate (the coverage aggregation run above) governs whether
the sweep's coverage is sufficient to draw a ranking across the design: pass (≥5 cells)
permits it; marginal (2–4) permits a ranking narrowed to the covered cells only, with the
uncovered cells named in the report; fail (<2) permits no ranking. §4's promotion row entry
condition "P7 complete" means the STEP gate; a promotion whose hypothesis ranges over an
uncovered cell additionally requires that cell's coverage, i.e. a re-run of the failed
tasks, not a re-run of the sweep.
**Scope: ~70 LOC of slurm + ~60 LOC of sweep args and plumbing + ~40 LOC aggregation script.**

**Step notes (2026-08-18, submitted; GPU gates pending):**
- Campaign `p7-stage2-h200` = `450f3583-c8dd-43f7-a59f-0642810746a5` (short `450f3583`). Exploration. Not `32d6574e`, not P6 `d6f44eab`.
- Probe flags on `profile_b1_flash_vs_autodiff_forces.py`: `--mode {dense,flash,both}`, `--pme`, `--grid-spacing`, `main(argv)`, PME via `eqx.tree_at` (`pme_alpha=0` / `pme_grid_points`). Frozen sidecar untouched.
- Array slurm: `--gres=gpu:h200:1`, `--time=00:15:00` from `3 * 0.557 * 20 / 3 = 11.14s` → 5 min + 10 min compile. `#SBATCH --partition=pi_so3` only: `pi_so3,mit_normal_gpu` plus H200 gres is an empty intersection (`sbatch: Requested node configuration is not available`). **myxcel submit** injects `--partition=mit_normal` unless `--partition pi_so3` is passed on the CLI.
- `scontrol show node node4009`: `Gres=gpu:h200:2` fully allocated → `%k=1`. Occupants: `19720353` (pi_so3, `gres/gpu:h200:2`, ~2 d remaining) and a mit_preemptable task. Scheduler `START_TIME` for the array ~ `2026-08-20T14:04`.
- Submitted from worktree `wt-20260807-132628`: array **20692608** (`0-11%1`, `CAMPAIGN_ID` exported); coverage **20692636** (`--dependency=afterany:20692608`). Task 12 (`prof_stage2_scope_ab.slurm`) is **not** submitted until the first array-task log exists.
- **Retarget 2026-08-18:** `scontrol show partition mit_preemptable` reports `gres/gpu:h200=194` vs `pi_so3`'s 2. `--test-only` with `pi_so3` in the partition list still binds to node4009 (start Aug 21 / Sep 11). Cancelled 20692608/20692636; resubmitted array **20705198** on **`mit_preemptable` only** (`0-11%1`, `Reason=Priority`) so the shared Hopper pool is eligible (`PreemptMode=REQUEUE`). Coverage **20705200** `afterany:20705198`.
- STEP / SIDECAR / `TERM_RANKING` / `trace.py` GPU legs wait on H200 start. Do not complete #4325 until those fire.

---

### P8 — Analysis/report step (where the contract actually fires)
**Cost: FREE.**

Create:
- `scripts/profiling/report.py` — loads records across stages, calls
  `assert_claim_supported` **before** emitting any ranked conclusion, and renders a
  stage-annotated bottleneck table. **Output surface (specified, because a gate that only
  asserts the contract raises would be satisfied by a report that renders nothing):** a
  GitHub-flavoured Markdown table written to stdout, and to a path given by `--out` when
  supplied. Columns, in order: `scope` | `exclusive_seconds` | `n_occurrences` |
  `pct_of_total` | `stage` | `n_atoms` | `platform` | `device_kind` | `attribution_method` |
  `probe_id`. Rows are sorted by `exclusive_seconds` descending. A scope whose value is
  `None` renders as `absent` in `exclusive_seconds`, never as `0.0`. **Mixed-attribution
  labelling (this is where P1's "must be labelled as mixed" requirement is discharged):** if
  the rendered ranking's rows do not all share one `attribution_method`, the table is
  preceded by the line `> MIXED ATTRIBUTION: this ranking combines named_scope and op_name
  attribution; per-row method is in the attribution_method column.` A footer line states the
  `contract_version`, `git_sha` and `xla_flags` the selected sources were unanimous on.
  **Gate addition:** a test asserts that a report rendered over a fixture set containing both
  attribution methods emits the MIXED ATTRIBUTION line, and that one rendered over a
  single-method set does not.

**Input surface.** Every probe writes its `ProbeRecord` to
`outputs/profiling/stage<N>/<probe_id>.json`; `report.py` discovers records by globbing
`outputs/profiling/stage*/*.json` and takes an optional explicit path list. P7's cluster
records reach that tree via `bth sync engaging --pull` followed by an rsync of
`outputs/profiling/` (state the command in the step notes); `report.py` never reads a remote
path.

**Gate:** running the report over the P3+P4 (Stage 0/1) records alone raises
`ClaimValidityError` on the term-ranking section; running it over P3+P4+P7 records succeeds
and produces the ranking. Both behaviours are asserted in `tests/profiling/` against
COMMITTED FIXTURE records under `tests/profiling/fixtures/records/` — CI cannot produce
Stage-2 GPU records. Because a fixture can silently drift from real producer output, the
fixtures are generated by `ProbeRecord.write` (never hand-authored) and a test asserts every
fixture round-trips through `ProbeRecord.read` without raising, so a producer/consumer
schema change fails the fixture test rather than passing silently.
**Scope: ~150 LOC.**

---

### P9 — CROSS-REPO: author the `jax-profiling` skill in `~/projects/xtrax`
**Cost: CROSS-REPO** (prose only; no execution). **This step edits a different repository.**

Create, in `/home/marielle/projects/xtrax` (**not** in this repo, **not** in
`~/.claude/skills/`):
- `agent_assets/skills/jax-profiling/SKILL.md` — TIER-1, self-contained, frontmatter with
  `name` / `description` / `triggers` / `jax_profiling_contract_version`, ending in a
  Workflow Index.
- `agent_assets/skills/jax-profiling/references/stages.md`
- `agent_assets/skills/jax-profiling/references/instrumentation.md`
- `agent_assets/skills/jax-profiling/references/claims.md`
- `agent_assets/skills/jax-profiling/references/loops.md`
- `agent_assets/skills/jax-profiling/references/failure-modes.md`

Follow §3 for content and `agent_assets/skills/using-xtrax/` for structure — read that
existing skill and at least two of its `references/` files first, and match them; do not
invent a layout.

**Why this is sequenced last despite being FREE to write.** The cost-ordering heuristic would
put a prose step first. Dependency ordering overrides it here: the skill asserts judgments
that P5, P6, and P7 *resolve empirically* — whether `named_scope` perturbs a fused hot path
(P5), what the real probe-to-target scale ratio is (P6), and whether the low-contention
partition is the one reputation predicts (P7 already inverted that assumption once). Writing
the skill before those land would encode three guesses as guidance, which is precisely the
skill-rot failure mode (pre-mortem story 3) manufactured on day one.

**Cross-repo obligations — flagged explicitly, because this is a real decision, not a
formality.** Editing another repo's files bypasses its review process unless we route through
it deliberately:
- Do the work on a branch in `~/projects/xtrax` and open a PR there. Do **not** commit to
  xtrax `main` from a prolix work session.
- Confirm xtrax's working tree is clean before starting; it has its own in-flight work and
  its own release cadence (currently `0.4.0a5`).
- Determine whether the skill also needs a `[[plugin.skills]]` entry in
  `~/projects/xtrax/.praxia/manifest.toml` (schema: `name` + `path =
  "agent_assets/skills/<name>/SKILL.md"`). **Open question, resolve during the step, do not
  assume:** that manifest currently declares only `[[plugin.workflows]]` and has *no* skills
  entry — yet `using-xtrax` exists and is in use anyway. So either the entry is optional and
  export is by directory convention, or `using-xtrax` is itself under-declared. Check which,
  and say which in the PR description.
- Nothing in P1–P8 may import from or depend on this skill. If a prolix gate would fail
  because the skill is absent, the dependency is backwards — fix the gate.
- Distribution is by xtrax's existing `just install-skills`; note in the PR that it
  `rmtree`+`copytree`s every skill directory, so the local `~/.claude/skills` tree is
  regenerated wholesale and must not be edited in place.

**Gate:** the skill directory exists at the path above with `SKILL.md` plus **all five**
`references/*.md` files named in the Create list. The floor is five, not three, because two
of them are load-bearing for criteria stated elsewhere: `failure-modes.md` carries the
skill-rot entry that §5's first cross-repo criterion names as the *only* mitigation for
`jax_profiling_contract_version` drift (which §3 concedes is otherwise unmitigated by
construction), and `claims.md` carries rules (b)–(d), which `SKILL.md`'s inline pattern
references rather than implementing (§3: "a copy of this pattern that omits them is not the
contract"). A skill shipping without either is missing a part §3 depends on, not merely
thinner than planned. `SKILL.md` frontmatter parses as YAML and carries all four fields; every
reference file opens with the `> Part of the ...` backlink line; a reader can answer "may I
cite this Stage-1 record for a term-ranking claim?" from `SKILL.md` alone without opening a
reference file; and no prolix-specific path or partition name appears as an instruction
(worked examples must be labelled as such). PR opened in the xtrax repo, not merged from
here.
**Scope: ~600 lines of prose across 6 files.**

---

### Post-P7 demotion (stated now so it isn't forgotten)

Once Stage 2 lands, Stage 0/1 probes are **demoted to regression tripwires** — cheap
structural checks that catch recompilation and dispatch-count regressions. They never block
a gate and are never cited for term ranking.

---

## 3. Skill outline — `xtrax/agent_assets/skills/jax-profiling/`

**Home (confirmed, corrected 2026-08-17):**
`/home/marielle/projects/xtrax/agent_assets/skills/jax-profiling/`. Not prolix; not
`~/.claude/skills/`. See §1 for why, and for why this does *not* mean the measurement code
moves too — it does not.

**Structure mirrors the existing `using-xtrax` skill in the same repo** — this is not an
invented layout. Read `agent_assets/skills/using-xtrax/SKILL.md` and its `references/`
(`tiling.md`, `run.md`, `training.md`, `cli.md`, `eda.md`, `sparse-distributed.md`,
`inference.md`) before authoring. The convention has three load-bearing properties:

1. **YAML frontmatter** with `name`, `description`, `triggers` (a list of concrete trigger
   phrases and symbol names, not a prose blurb), and a version-pin field. `using-xtrax` uses
   `xtrax_version: 0.4.0a5`; this skill pins the *methodology*, not the library, so it uses
   `jax_profiling_contract_version` (the `scripts/profiling/claims.py` contract version it
   describes) — a differently-named field for a genuinely different pin, rather than a
   misleading reuse of `xtrax_version`. **Drift is unmitigated by construction and is
   accepted, not solved.** The skill lives in a repo with no dependency on prolix, so no
   automated check can compare the pinned string against `claims.py`'s `CONTRACT_VERSION`.
   The mitigation is procedural and one-directional: a major bump of `CONTRACT_VERSION`
   obliges a follow-up PR to the skill, and the skill's `references/failure-modes.md`
   skill-rot entry names a stale `jax_profiling_contract_version` as its own tell. A reader
   who finds the pinned version behind the version stamped in a record they are holding
   should distrust the skill's claim rules and re-read `claims.py`.
2. **TIER-1 `SKILL.md` is self-contained** — it must be usable without loading any reference
   file, and ends in a *Workflow Index* mapping a task to the one or two reference files it
   needs.
3. **TIER-2 detail lives in `references/*.md`, loaded on demand, one file per topic** — never
   one flat file. Each reference file opens with the same backlink line the existing skill's
   reference files use, adapted to this skill:

```
> Part of the `jax-profiling` skill (`agent_assets/skills/jax-profiling/SKILL.md`) — TIER-2 deep reference.
```

The content below warrants the split: the stage/roofline/scale rules are the self-contained
core, while tooling mechanics, instrumentation procedure, and cluster discipline are each
deep enough to be loaded only when a task reaches that layer.

### `SKILL.md` (TIER-1 — self-contained)

- **Frontmatter** — `name: jax-profiling`; `description` naming the staged model and the
  claim-class contract; `triggers` including *"why is this JAX code slow"*, *"failed
  throughput gate"*, *"suspected recompilation / dispatch regression"*, `cost_analysis`,
  `named_scope`, `jax.profiler.trace`, `ProbeRecord`, `ClaimClass`, `assert_claim_supported`,
  *"can I cite this profiling number"*; `jax_profiling_contract_version`.
- **When to use / when not to** — performance questions only. Explicitly *not* for
  correctness debugging or NaN hunting.
- **The four stages, and what each can and cannot prove** — the stage table (0 = free
  structural, 1 = cheap CPU dispatch/bonded, 2 = GPU small-system term ranking, 3 = GPU
  end-to-end), with the *cannot* column carrying equal weight to the *can* column.
- **The roofline rule** — why FFT (bandwidth-bound, ~0.3–0.5 FLOPs/byte) and dense GEMM
  (compute-bound, 1.5–3.5+) reorder between CPU and GPU, and why that makes sub-Stage-2 term
  ranking structurally invalid rather than merely noisy.
- **The scale rule** — why platform validity does not imply scale validity; the
  ~1-order-of-magnitude extrapolation limit; when a mid-scale rung becomes mandatory.
- **Minimal working pattern** — the smallest complete loop: emit a stamped `ProbeRecord`,
  call `assert_claim_supported`, read the verdict. Runnable as written, which means
  **self-contained**: the pattern includes the ~15-line reference implementation of both
  primitives inline. It must open with the sentence "`ProbeRecord` and
  `assert_claim_supported` are a *pattern*, not a library — xtrax ships no profiling API,
  and this skill deliberately does not depend on one. Copy the pattern into your repo, or
  point at your own implementation." A prolix implementation exists at
  `scripts/profiling/claims.py` and may be cited only as a labelled worked example. The
  inline pattern must implement, at minimum: the stage and `n_atoms` fields, the stage≥2 ⇒
  GPU construction check, and rule (a), the `TERM_RANKING` Stage-2 floor — the one rule prose
  cannot enforce and the reason the contract exists. Rules (b) fail-closed target, (c) scale
  ratio, and (d) provenance unanimity are stated normatively in `references/claims.md` and
  are referenced from the pattern by an inline comment (`# rules (b)-(d): see
  references/claims.md` — a copy of this pattern that omits them is not the contract). The
  ~15-line budget applies to the demonstration, not to the contract; if the minimum above
  exceeds it, the budget yields.
- **Workflow Index** — task → reference file(s), following `using-xtrax`'s numbered list.

### `references/stages.md` — tool selection per stage

Which question, which tool: `cost_analysis()` (structural, no execution) vs `named_scope` +
`jax.profiler.trace` (attribution) vs wall-clock A/B (end-to-end) vs regression-based
compile/compute decomposition. What each stage's records may and may not be cited for.

### `references/instrumentation.md` — instrumenting a hot path safely

Adding `named_scope`; the **mandatory perturbation check** (A/B total step time with scopes
on vs off, ±5% median with a 90% percentile-bootstrap CI inside the margin — ≥21 interleaved
A/B pairs, `B=10,000` resamples, seeded — checked both whole-step and
per-instrumented-function); the fused-`scan`/tile-reduction caveat; the trace-level
(op-name) fallback when scopes perturb fusion; and the **two-failure-mode distinction** —
`|median(d_i)| > 0.05` is a DETECTED perturbation (revert the scopes), while a CI that
misses the margin with `|median(d_i)| <= 0.05` is INSUFFICIENT POWER (increase `n_pairs` from
a pilot IQR estimate and re-run; never revert on this branch alone). This is the procedural
counterpart to step P5.

### `references/claims.md` — the claim-class contract

`ClaimClass` semantics (`STRUCTURAL`, `DISPATCH_COUNT`, `TERM_RANKING`, `END_TO_END`); the
stage and `n_atoms` stamps; why the contract **raises** rather than warns; what makes a
number paper-citable; how to read a `ClaimValidityError` and what the two legitimate
responses are (get a better record, or weaken the claim — never loosen the limit).

### `references/loops.md` — discovery loop vs confirmation loop, and GPU budget discipline

Ungated exploratory probes for iteration; exactly one pre-registered experiment with an
anchored threshold per surviving hypothesis; why discovery must not be gated (it was never in
the confirmation tool's remit). Then cluster discipline: check *live* contention before
choosing a partition (reputation lies — see P7's inverted assumption), sweep many hypotheses
per submission, small system before large.

### `references/failure-modes.md` — known failure modes

Bottleneck-found-in-week-one over-build; unrepresentative probe scale (the pre-mortem's
story 2); and skill rot conferring false authority — the one enforced rule keeps passing
while the dozen unenforced judgments silently go stale. Each entry names its tell.

> **Prolix-specific detail does not belong in this skill.** Concrete paths
> (`scripts/profiling/`, `flash_explicit.py`), the `pi_so3`/H200 partition names, and the
> DHFR 23,558-atom target are prolix facts and stay in *this* spec. The skill may use them as
> clearly-labelled worked examples, never as instructions. This is the direct mitigation for
> the path-rename half of pre-mortem story 3.

---

## 4. Follow-ups (backlog-item-shaped, out of scope for this plan)

| Title | Description |
|---|---|
| **Stage 3: DHFR-scale end-to-end profiling validation** | Run the surviving Stage-2 hypotheses at 23,558 atoms on H200 to confirm the bottleneck ranking holds at production scale. Entry condition: P7 complete + P6 scale-ratio check passed. |
| **Promote surviving hypotheses to pre-registered bathos experiments** | One `scripts/experiments/` experiment + sidecar per hypothesis with an anchored threshold (e.g. "removing X yields ≥1.5× on the DHFR flash path"). Entry condition: P7 complete. |
| **Mid-scale probe rung (conditional)** | Build a ~8–10k-atom solvated probe if P6 finds `23558 / n_atoms > 10`. Blocking for any Stage-3 claim if triggered. |
| **Upstream `scripts/profiling/` *code* to `xtrax.profiling`** | Cross-repo (`~/projects/xtrax`), requires PR + alpha release + a prolix dependency bump to consume it. **Deferred behind the S8 trigger**: a second consumer repo asks, OR the API survives two complete hunt loops unchanged. Declined, not pending, absent a trigger. **Distinct from P9** — see the note below; P9 shipping does not fire this trigger. |
| **xtrax cost-estimator empirical calibration** | Explicitly deprioritised — identified in the brainstorm as proximity-pulled roadmap creep. Stage 0 is disqualified from cross-platform magnitude claims by construction, so better calibration cannot answer the DHFR question. |
| **bathos discovery-mode campaign type** | Not needed and arguably a category error (S9). Recorded only so the idea is not re-raised without the counterargument. |
| **Named-scope coverage for the neighbor-list rebuild path** | `simulate.py` / NL rebuild is an untested-but-plausible contributor to the 10× gap and currently has no scope coverage. |
| **CI regression tripwire from Stage 0/1 probes** | Wire the demoted Stage 0/1 probes into fast CI to catch recompilation and dispatch-count regressions. Entry condition: post-P7 demotion. |
| **Re-evaluate `titanix` as a GPU probe target** | Disqualified today solely because JAX is not installed; installing it would open a genuinely zero-queue small-probe target. |
| **Land the `jax-profiling` skill PR in xtrax** | P9 opens a PR; this row tracks it to merge or decline. Reviewer: xtrax maintainer. Checklist: §5's cross-repo criteria. Suggested reviewer aid: run `plugin-dev:skill-reviewer` over `SKILL.md` before requesting review. **Accepted risk, stated:** xtrax CI (`ci`/`docs`/`publish`/`audit-judgment` workflows) validates no skill content, structure, or frontmatter, so the §5 criteria are the only check and they are manual. **If declined or unmerged:** nothing in P1–P8 changes — the skill is judgment-layer only and no prolix gate depends on it (P9, §5) — and the methodology remains recorded in this spec's §3, which is the fallback home. Do not re-file it in prolix as a workaround. |

### Are P9 and "upstream `scripts/profiling/` to `xtrax.profiling`" the same follow-up?

**Decision: genuinely independent. Two rows, not one.** Stated explicitly because the shared
destination repo makes them look like one item, and the previous revision's collapse of
"xtrax" into a single concept is exactly the error §1 corrects.

They differ on every axis that matters:

| | **P9 — skill (in scope, this plan)** | **Upstream code (follow-up, deferred)** |
|---|---|---|
| Artifact | Prose agent asset | Python package |
| Path | `agent_assets/skills/jax-profiling/` | `src/xtrax/profiling/` |
| Import surface | None | Public API |
| Release needed | No | Alpha release + prolix dependency bump |
| Trigger | Unconditional (P9 is a step) | S8: second consumer, or two clean hunt loops |
| Breaks on rejection | Nothing in P1–P8 | Nothing in P1–P8 |

**They are, however, *related* in one direction only, and that relation is worth naming.**
P9's existence weakly raises the eventual probability of the code upstream — a JAX-general
methodology living in xtrax makes xtrax the natural home for the primitives that implement
it, and a second consumer is likelier to find the skill than to find prolix's
`scripts/profiling/`. That is a *forecast about the S8 trigger's likelihood*, not the trigger
firing. Concretely: **do not treat P9 as evidence the S8 trigger fired, and do not upstream
the code just because the skill is already there.** The trigger's terms are unchanged; if the
proximity of the skill starts pulling at the code, that is the roadmap creep the brainstorm
already flagged and rejected once.

---

## 5. Acceptance criteria (Phase 1 / steps P1–P9)

- [ ] No file at any depth under `scripts/profiling/` imports `prolix` (AST-asserted, recursive, in CI).
- [ ] `ProbeRecord` carries `stage` **and** `n_atoms`; both are required fields.
- [ ] `ProbeRecord` rejects `stage >= 2` on a non-GPU platform at construction.
- [ ] `assert_claim_supported` **raises** (does not warn) on a sub-Stage-2 term-ranking claim.
- [ ] `assert_claim_supported` **raises** on an end-to-end extrapolation exceeding 10× in `n_atoms`.
- [ ] `assert_claim_supported` **raises** on an END_TO_END claim that declares no target system — the scale guard is fail-closed, not opt-in.
- [ ] `named_scope` labels cover bonded, LJ-direct, Coulomb-direct, PME-reciprocal, and the grad call on the DENSE path; on the FLASH path they cover bonded, the fused LJ+Coulomb-direct tile reduction, and the grad call, with PME-reciprocal covered by `pme.py`'s pre-existing `pme_*` labels (`flash_explicit.py`'s nonbonded path routes the reciprocal through `make_spme_energy_fn`). Flash LJ-direct and Coulomb-direct are NOT separable at `named_scope` granularity: they are accumulated in one fused `jax.checkpoint` tile body and splitting them would restructure the tile reduction, violating P2's no-logic-change guarantee and inducing exactly the fusion perturbation P5 forbids. Flash LJ-vs-Coulomb attribution is available only at op-name granularity; if it is ever needed, the restructuring is separate scoped work with its own perturbation gate.
- [ ] The P5 perturbation check passes, or the offending scope GROUP is identified by the bisect procedure, reverted, and the fallback recorded.
- [ ] `trace.py` reproduces exclusive per-scope times on a synthetic trace with known ground truth.
- [ ] `claims.py` defines `CONTRACT_VERSION` and every emitted record serialises the version in force when it was emitted.
- [ ] `ProbeRecord` auto-DEFAULTS precision, JAX/jaxlib versions, and `XLA_FLAGS` from the environment; `assert_claim_supported` refuses to rank across records that differ in precision, XLA flags, device kind, or `git_sha`.
- [ ] `profile_b1_flash_vs_autodiff_forces.py` completes a CPU smoke run and reports actual `n_atoms`.
- [ ] P7 produces Stage-2 GPU records from a single sweep submission on `pi_so3`, campaign-associated in bathos.
- [ ] The perturbation check is replicated at Stage 2 on GPU, or per-scope attribution from P7 is recorded as unavailable and the op-name fallback used.

### Cross-repo criteria (P9 — `~/projects/xtrax`)

Kept in a separate subsection because they are satisfied in a **different repository** and
cannot be verified by any prolix gate or CI job.

- [ ] `SKILL.md`'s `jax_profiling_contract_version` pins the same string as `claims.py`'s `CONTRACT_VERSION` at authoring time. Unverifiable by any prolix gate or CI job by construction (§3: the skill lives in a repo with no prolix dependency). The obligation is procedural and one-directional: a MAJOR `CONTRACT_VERSION` bump obliges a follow-up PR to the skill, and `references/failure-modes.md` names a stale pin as its own tell.
- [ ] The skill's *source of truth* is `/home/marielle/projects/xtrax/agent_assets/skills/jax-profiling/SKILL.md` — authored and reviewed in xtrax, not in prolix and not hand-authored directly in `~/.claude/skills/`. A copy at `~/.claude/skills/jax-profiling/` produced by xtrax's own `just install-skills` (`scripts/install_skills.py`, which `rmtree`+`copytree`s every directory under `agent_assets/skills/`) is the sanctioned distribution artifact — identical treatment to the sibling `using-xtrax` — and does not violate this criterion.
- [ ] `SKILL.md` frontmatter parses as YAML and carries `name`, `description`, `triggers` (list), and `jax_profiling_contract_version`, matching the field pattern of the sibling `using-xtrax` skill.
- [ ] TIER-2 content is split across all five files named in P9's Create list, including `claims.md` and `failure-modes.md`, each opening with the `> Part of the ...` backlink line — not one flat file.
- [ ] `SKILL.md` alone answers "may I cite this record for this claim class?" with no reference file loaded.
- [ ] No prolix-specific path, partition name, or atom count appears in the skill as an instruction; any that appear are labelled worked examples.
- [ ] `SKILL.md` states explicitly that the primitives are a copyable reference pattern and that xtrax ships no profiling API — a reader must not be able to conclude the skill documents an importable xtrax surface.
- [ ] The PR names a reviewer and links §5's cross-repo criteria as its checklist; the §4 tracking row is filed.
- [ ] The change is delivered as a **PR in the xtrax repo**, on a branch, against a clean xtrax working tree — not committed to xtrax `main` from a prolix session.
- [ ] The `[[plugin.skills]]` manifest question is resolved and the answer stated in the PR description (currently `~/projects/xtrax/.praxia/manifest.toml` declares workflows only, yet `using-xtrax` ships without an entry — determine which is correct rather than assuming).
- [ ] **Separation preserved:** no file under `scripts/profiling/` has moved to xtrax, and nothing in P1–P8 imports from or is gated on the skill. Landing the skill did not fire the S8 code-upstream trigger.
