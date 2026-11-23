---
title: 'prolix CLI and MCP server: quick in-process operations, long runs rendered as jobs'
description: 'Implementation spec for debt #3555: a cisternal-wired prolix CLI and MCP server (src/prolix/agent) exposing seconds-scale operations in-process on CPU and rendering or submitting long runs through myxcel; built on xtrax.agent.'
status: draft
task_id: 261123_xtrax-entrypoint-cli-mcp
date: '261123'
---
# prolix CLI and MCP server (debt #3555)

Base: prolix `origin/main` 5ce098d. Builds on the converged `261123_xtrax-agent-entry.md` (xtrax PR #284: `xtrax.agent.run_cli`
/ `serve`; its §7 lists what this spec owns). Owner decision (fixed): only quick operations run in-process (latency is stated honestly in §4); long
runs are rendered as a job script or submitted via myxcel, never run inside a call.

## 1. Problem, and what recon got wrong

prolix has no CLI, `__main__` or `[project.scripts]`; `src/prolix/viewer_server.py:268` is its only argparse. Reading code
changes the design in four ways:

1. **`MolecularBundle.from_pdb` cannot be the loader.** It validates `forcefield` and then ignores it; the parmed path yields
   empty parameters for a PDB, as its own comment says (`types/bundles.py:277-297,336-345`). Its docstring ("AMBER14 parameters") is false.
2. **The working loader is unpackaged**: `scripts/benchmarks/_b1_paramize.py:41` (proxide `parse_structure(parameterize_md=True)`,
   ff19SB, `ExclusionSpec.from_protein`); `packages.find where=["src"]` (pyproject.toml:50-51).
3. **`make_bundle_from_system` raised `NameError`** (`use_size_buckets`, `physics/system.py:874-876`); fixed in maraxen/prolix#46, a
   prerequisite of this PR (C17). **A second `NameError`, `_chunked_lj_energy_nl_core` (`physics/optimization.py:363`), is prolix debt #3586**: it
   sits on the stacked neighbor-list path (`neighbor=` bound, `EnsemblePlan`/NL kernels). All seven tools use the dense path
   (`energy_fn_from_bundle` with no `neighbor`) and never construct a neighbor list; a test asserts `neighbor` is never passed.
4. **`run_minimization` ignores `steps`, `dt_start` and FIRE**: only `sd_steps` steepest-descent iterations run
   (`physics/simulate.py:141-197`). `run_nve`/`run_nvt_nose_hoover` have no seed (key 0) and return positions only (`:259-328`),
   so a run cannot be chunked or resumed.

## 2. Decisions (recommended; rejected alternatives)

| # | Decision | Rejected |
|---|---|---|
| D1 | Seven tools, registry `prolix`, one body per tool, wired to CLI and MCP by `cisternal.wire` | hand-written CLI+MCP pair (drifts); generic `run_script` tool |
| D2 | In-process tools run on CPU only, Pallas off, x64 at library default; GPU work is a job | selectable GPU in-process (contention, Blackwell quirks, untestable in CI) |
| D3 | `render_job` wraps a repo script in an sbatch file and returns the exact `myxcel submit` argv; `submit_job` runs it, dry-run unless `confirm=true` | a prolix long-run executor (no checkpoint API); submit by default; `myxcel submit-job PRESET COMMAND` (no reviewable script) |
| D4 | `out_dir` is a required argument for in-process tools; `render_job` writes inside the checkout (see §6) | default `outputs/` or cwd |
| D5 | Copy the b1 loader into the package, with a parity test against the script | import from `scripts/`; refactor the script to call the package |
| D6 | Base `xtrax>=0.4.1b2,<0.5` unchanged; only the `agent` extra pins `xtrax[plugin]>=X,<0.5`; `check_xtrax_pin.py` reads the base floor from pyproject (regex tolerant of extras) | raising the base floor for every library user; two hard-coded floors |
| D7 | MCP tool wrappers are `async` and run the op via `asyncio.to_thread` under an `asyncio.Lock`-guarded semaphore of 1; a second concurrent call refuses immediately (`busy`) | sync bodies (cisternal runs them on the event loop, A16); queueing silently |
| D8 | `submit_job` is disabled unless `PROLIX_AGENT_ALLOW_SUBMIT=1` is in the server/CLI environment (set by the human, out of band) | `confirm` flag as the only gate (an agent can set it) |
| D9 | CI installs `--extra dev --extra agent`; ops take an optional pre-built `bundle=` seam so tests skip the 13 s load | importorskip everywhere (everything skips in CI) |

## 3. Layout and entry points

```
src/prolix/agent/__init__.py   require_agent_extra() only (aminx/agent/__init__.py pattern); imports nothing heavy
src/prolix/agent/errors.py     AgentError > InputError, RefusalError
src/prolix/agent/_runtime.py   ensure_cpu(), cap constants, run lock, versions(), sha256(), atomic_write(), write_record()
src/prolix/agent/system.py     load_system_bundle(path, ff_name, periodic): copy of _b1_paramize.paramize_pdb_to_bundle
src/prolix/agent/ops.py        pure functions, one per operation (no cisternal import); golden tests call these
src/prolix/agent/render.py     render_job_script(), submit_argv() (no cisternal import)
src/prolix/agent/tools.py      TOOL_NAMES; sync tools in registry "prolix" (CLI) and async to_thread twins in "prolix_mcp" (MCP), same names
src/prolix/agent/cli.py        require_agent_extra() on its FIRST line; cyclopts app; wire(None, app, ...); main()
src/prolix/agent/mcp.py        require_agent_extra() on its first line; build_server(); main()
```

`pyproject.toml`: `[project.scripts] prolix = "prolix.agent.cli:main"`, `prolix-mcp = "prolix.agent.mcp:main"`;
`agent = ["xtrax[plugin]>=X,<0.5"]` (cisternal, cyclopts, fastmcp come transitively); base `xtrax>=0.4.1b2,<0.5` is unchanged. `X` = first
xtrax release containing `xtrax.agent` (after 0.4.1b9; A1). `cli.py` and `mcp.py` call `require_agent_extra()` before any cisternal,
cyclopts or fastmcp import (they wire at import time), so a missing extra prints the `prolix[agent]` hint instead of a
`ModuleNotFoundError`. CI (`ci.yml:36,59`) today runs `uv sync --extra dev` and `uv run ty check`: Task 1 changes both steps to `--extra dev --extra agent`, so the cisternal imports type-check and agent tests run rather than skip (C4). `prolix/__init__.py` does not import `prolix.agent`; `ops/render/_runtime` import none of the three. Heavy imports
(`jax`, `proxide`, `prolix.api`) sit inside operation bodies. `import prolix` costs ~13 s cold (A9), so `prolix --help` pays it;
a lazy `prolix/__init__` is a follow-up.

```python
def main() -> None:                 # cli.py, after the require/import lines
    _runtime.ensure_cpu()           # FIRST: run_cli starts and dispatches in one call, so nothing can run between
    agent.run_cli(app)              # exit 2 bad cache config, exit 1 missing xtrax[plugin]
def main() -> None:                 # mcp.py
    _runtime.ensure_cpu(); agent.serve(build_server())
```

`build_server()` mirrors `aminx.agent.mcp.build_server`: `FastMCP("prolix", instructions=...)`, `cisternal.wire(server, registry="prolix_mcp",
expected=list(TOOL_NAMES))` (the async twins, D7: cisternal's CLI path is sync, `wired.py:103`, so one async function cannot serve both), `add_middleware(CisternalMiddleware(adapter=PassthroughAdapter(), reraise=True))`. `cli.py`:
`wire(None, app, registry="prolix", expected=TOOL_NAMES, cli_contract=CONTRACT)` at import, so a renamed or dropped tool fails at
import (demistify `cli.py:1013`). `CONTRACT`: `exit_codes={InputError: 4, RefusalError: 3}`, `options=[json_option()]` and a `format_success(result, ctx)` that writes
`json.dumps(result, sort_keys=True)` to stdout when `ctx.options["json_out"]` else `key: value` lines; `json_option` alone changes nothing (C5). Nothing else may
write to stdout; logs go to stderr. Tool bodies return dicts and
raise `AgentError`; they never print (cisternal rule 2).

**Exit codes (published, tested):** 0 success; 1 `xtrax[plugin]`/`prolix[agent]` missing (xtrax.agent, `require_agent_extra`); 2 bad jax-cache
configuration (xtrax.agent) and cyclopts usage errors (cyclopts' default on a bad argument, A17 confirms in Task 4 by test); 3 `RefusalError` (caps, busy,
submit disabled); 4 `InputError`; 1 is also an unhandled exception's default traceback status. The table lives in `--help` epilog and a test per row.

## 4. Operations

In-process tools take `out_dir` (required) and write `<tool>.json` (run record, §5) there, plus named side files. Arrays never go
inline: they are written to `.npz` and the result carries path and sha256. Shared arguments: `pdb`, `ff` (default
`protein.ff19SB.xml`, a bundled proxide name or an absolute path, as `_resolve_ff_path` does), `overwrite=false`.

| Tool | Does | Library call | Side files | Result keys |
|---|---|---|---|---|
| `load_system` | parameterise, validate, summarise | `system.load_system_bundle` | none | real/padded atoms, bucket idx, term counts, charge sum, `has_pbc`, ff path |
| `inspect_params` | parameter rows for one term (`bond,angle,dihedral,improper,charge,lj`), `limit<=200` | same bundle | none | rows, `truncated` |
| `energy_breakdown` | total, bonded, nonbonded, force stats | `energy_fn_from_bundle` (total); `bonded_energy_fn_from_bundle` = bond+angle+dihedral+improper(+CMAP if present) and no LJ, Coulomb or 1-4 terms (`bundle_md.py:223`; Task 2 records the exact term list in the docstring from a read); nonbonded = total - bonded, includes 1-4 | none | `total`, `bonded`, `nonbonded_by_difference`, median/max force (kcal/mol/Å) |
| `minimize` | bounded steepest descent (`steps` is not passed: unused, A4) | `run_minimization(..., sd_steps=n)` | `minimized.npz` | energy before/after, `sd_steps`; refuses to report success unless after <= before |
| `md_smoke` | short NVE or NVT Nose-Hoover; arguments `steps`, `dt_fs`, `ensemble`, `temperature_k` | `run_nve` / `run_nvt_nose_hoover` with `dt = dt_fs / AKMA_TIME_UNIT_FS` (`prolix/simulate.py:56`; AKMA units, `physics/simulate.py:283,310`) | `final.npz` | `finite`, start/end energy, steps, dt, ensemble, `seed:"fixed 0"`; no conservation claim |
| `render_job` | write an sbatch script (§6) | `render.render_job_script` | `job.slurm` | path, sha256, resources, `submit_argv` |
| `submit_job` | `myxcel submit` (§6) | `render.submit_argv` + `subprocess.run` | none | `dry_run`, argv, exit code, stdout, stderr |

Positions: `api.bundle_md.active_positions(bundle)` (real atoms), as `_b1_paramize.bundle_force_stats` does; masses `bundle.masses[:n_real]`.

**Latency, stated honestly:** a cold call costs about 13 s to parameterise plus about 13 s to compile the energy function, and each call builds new
closures, so every call recompiles (A8). Operations are *quick*, not seconds-scale. `ops.py` keeps a per-process cache of at most 2 loaded bundles keyed by
`(sha256(pdb), ff, sha256(ff file))`, and of energy/force closures per bundle key, so repeat calls in one MCP server skip load and compile; the CLI pays
the cost per invocation. The record carries `bundle_cache: hit|miss`. Every op takes `bundle=None` (a pre-built bundle; tests and library callers).

**Hard caps** (constants in `_runtime.py`; a call may lower, never raise; no CLI override). Placeholders, which Task 2 must lower until
the largest allowed combination runs in about 60 s on 8 CPU threads (dense energy scales ~N², so measure at the cap, not at 327 atoms):
`MAX_ATOMS=1000`, `MAX_MIN_STEPS=200`, `MAX_MD_STEPS=500`, `MAX_DT_FS=2.0`. The atom check counts `ATOM`/`HETATM` lines before
parameterising and checks `n_real` again after. A violation raises `RefusalError` naming the limit, the request, and `render_job`.
`InputError` for: missing file, non-`.pdb`, unknown ff name, existing record without `overwrite`, `out_dir` outside an allowed root, `dt_fs` above cap or not positive,
non-finite input energy,  and a loaded bundle whose `bond_params` are all zero (the `from_pdb` failure mode).
An H-free PDB is **warned, not refused**: the loader adds no hydrogens (both fixtures, 1CRN and `mini.pdb`, load as heavy-atom-only with 327 and 20 atoms, A8) and refusing would reject the CI fixture; every result carries `warnings: ["no hydrogens: energies are for a heavy-atom model"]` and the record stores `n_hydrogen`. Refusals precede any write. `periodic` is not offered: the copied loader sets no box for periodic bundles (it would get a silent default box), so
systems are always `free` boundary (C10). `out_dir` must resolve (symlinks followed) under `$PROLIX_AGENT_OUT_ROOT` if set, else under cwd; `..` escapes
refuse. Side files are written to `<name>.tmp.npz` (a `.npz` suffix, so `np.savez` does not append one) then `os.replace`d.

**Concurrency (D7, A16):** cisternal runs a sync tool body directly on the event loop (compose.py:130-133 via `shim.dispatch`, shim.py:62-64), so a 25 s
op would freeze the server. MCP wrappers in `tools.py` are `async def` and call the op through `asyncio.to_thread`; a module-level `asyncio.Lock`
admits one op at a time and a second concurrent call raises `RefusalError("busy")` at once (`lock.locked()` check, no waiting). The CLI path calls
the sync op directly. Test: start one op blocked on a `threading.Event`, call a second, assert `busy` and that the event loop stays responsive
(a concurrent `asyncio.sleep(0)` ticker keeps advancing), then release and assert the first completes.

## 5. Run record and atomic writes

`<out_dir>/<tool>.json`, schema `prolix.agent.run/1`: tempfile in `out_dir` then `os.replace`; side files first, record last, so a record
implies its side files.

```
schema, tool, started_utc, wall_s, args (paths resolved), caps {name: value},
inputs [{path, sha256, bytes}], outputs [{path, sha256, bytes}],
versions {prolix, xtrax, jax, jaxlib, proxide, parmed?, python, cisternal?}   # importlib.metadata (A11)
jax {backend, x64, devices, pallas}
jax_cache  xtrax.jax_cache.current().as_record() or {"configured": false}     # public accessor, A7
git {rev, dirty} when the source is a checkout, else null
result {...}
```

## 6. Job rendering and submission

Today (read): `scripts/slurm/*.slurm` carry `#SBATCH` headers (`b1_init_exec.slurm:2-10`: `mit_preemptable`, `gpu:h200:1`, 12:00:00,
relative `outputs/logs/slurm/` logs), `source scripts/slurm/_common_env.sh` (cd to repo root, lock-keyed uv venv via `_uv_env.sh`
under myxcel), then `uv run python scripts/...`. `myxcel.toml` has presets and `[uv] venv = "lock-keyed"`. `myxcel submit REMOTE PROJECT
SCRIPT_PATH` (flags `--dry-run`, `--respect-directives`, `--array`, `--partition`, `--time`) requires the script to exist locally and
translates a path **inside the project's local tree** to the matching remote path; a script outside it is passed verbatim and must
already exist on the remote (myxcel `cli.py:1792-1830`, A12). No prolix executor exists for production MD, ensembles, REMD or fitting:
they are ad hoc scripts, with no checkpoint or resume API.

**Contract.** `render_job(script, args, preset, time, name, repo_root=None)`:
- `script`: repo-relative path under `scripts/`, `.py`, existing, resolving inside `repo_root` (`..`, absolute, symlink escapes refused), **and listed in
  `src/prolix/agent/job_scripts.txt`** (an allowlist file in the package; initial content the scripts a human names in the PR; empty list means none renders).
  `repo_root` defaults to the nearest ancestor of cwd with `pyproject.toml` `name = "prolix"` and `scripts/slurm/_uv_env.sh`, else `InputError`.
- Artifact: `<repo_root>/outputs/agent_jobs/<name>/job.slurm` and `render_job.json` (script sha256, source script sha256, args, preset, time, git rev).
  Re-rendering an existing `<name>` refuses (`InputError`), so a rendered script is immutable after review. Writing inside the checkout is what lets
  `myxcel submit` map the path; the one exception to D4.
- `args`: strings, each `shlex.quote`d. `preset` closed table: `cpu_quick` (`#SBATCH --partition=mit_quicktest --time=0:10:00 --cpus-per-task=2 --mem=4G`, the
  headers of `l3_smoke.slurm:2-9`; `mit_quicktest` is a partition used in repo scripts, not a myxcel.toml preset) or `gpu_h200_preempt` (`mit_preemptable`,
  `--gres=gpu:h200:1`, 8 cpus, 256G; `b1_init_exec.slurm:2-8`). `time` `HH:MM:SS`, default and maximum `12:00:00`. `name` `[A-Za-z0-9_-]{1,40}`.
- Template (golden-tested; the header sources are those scripts actually use: `l3_smoke.slurm:34-35` and `b1_init_exec.slurm:111-115`, not `_common_env.sh`, whose
  `write_engaging_manifest.py` call is a side effect we do not want):
  ```
  #!/bin/bash
  <preset #SBATCH lines> --job-name=prolix_<name> --output/--error=outputs/logs/slurm/%j.<name>.{out,err}
  set -euo pipefail
  cd "${SLURM_SUBMIT_DIR:?}"
  mkdir -p outputs/logs/slurm
  source scripts/slurm/_uv_env.sh && prolix_uv_sync
  rc=0
  uv run python <script> <quoted args> || rc=$?
  echo "=== <name> exit=${rc} ==="
  exit "${rc}"
  ```
  `|| rc=$?` keeps the status line reachable under `set -e` (a bare `echo $?` after the command never runs on failure; the golden test also runs the template
  with a stub `uv` that fails and asserts the line prints and the script exits non-zero). No `bth run`.
- Result: `artifact` (absolute path of `job.slurm`), `submit_argv`, `notes` (sync the checkout to the remote, then submit).

`submit_argv = ["myxcel","submit",<remote>,"prolix",<absolute artifact path>,"--respect-directives"]`, executed with `cwd=repo_root`. myxcel resolves the script
path and maps it to the remote only when it lies under `profile.local_workspace/<project>` or the detected worktree root (`cli.py:1793-1820`); a render from a
checkout that is not that tree gets myxcel's own "script is outside ..." warning, so `notes` states that `repo_root` must equal the myxcel project root. Test: a stub `myxcel` records argv and cwd; assert the artifact path is absolute, exists, and sits under `repo_root`.

`submit_job(name, remote="engaging", confirm=false, repo_root=None)`:
- Threat model: the caller is an LLM agent that can set any tool argument and read tool output, but cannot set the server's or CLI's process environment. So `confirm`
  is a typo guard only; the authorisation is **`PROLIX_AGENT_ALLOW_SUBMIT=1` in the process environment** (D8). Unset: any non-dry-run attempt raises `RefusalError`;
  dry-run still works. Residual risk: an agent that can edit the human's shell profile or launch its own server; out of scope, named in the PR.
- Refuses unless `render_job.json` exists, the artifact's sha256 equals the recorded one (edited scripts do not submit), and the recorded source-script sha256 still
  matches the file in `repo_root`. Appends `--dry-run` unless `confirm`; runs `subprocess.run(argv, shell=False, timeout=120, capture_output=True, cwd=repo_root)`; returns argv,
  exit code, stdout, stderr; parses no job id. Missing `myxcel` raises `InputError` with the argv for a human to run. `subprocess` is spawn-safe; cisternal forbids only `fork` multiprocessing.

## 7. Process, device and multiprocessing policy

- Startup is `xtrax.agent` only; `main()` never calls `jax_cache.configure()` or `cisternal.init()` itself.
- `ensure_cpu()`: `jax.config.update("jax_platforms","cpu")`, first statement of both `main()`s (A6), never edits `os.environ`. `tests/agent/conftest.py` calls it in a session-scoped autouse fixture before any test touches JAX, so a GPU host (titanix) does not make every in-process test refuse (C9). Each in-process
  op re-asserts `jax.default_backend() == "cpu"` and raises `RefusalError` if a backend was already initialised elsewhere (fails closed; a
  GPU box cannot silently run an in-process tool on GPU). `xtrax/jax_cache.py` has no `devices(`/`default_backend`/`xla_bridge` call (A6);
  a subprocess test checks that `import prolix` leaves `jax.lib.xla_bridge.backends_are_initialized()` false. Pallas stays default-off
  (`pallas_kernels.py:81-92`, enabled only by `PROLIX_USE_PALLAS`); the record stores the flag. x64 is untouched (tests flip it per fixture,
  `tests/physics/conftest.py:66-80`). No `device=` argument is offered.
- No `multiprocessing`: `src/` has none (A15); a test greps `src/prolix/agent` for `multiprocessing`/`os.fork`.
- `MAX_ATOMS` bounds the dense `N x N` exclusion matrices (`bundle_md.py` docstring: ~4.4 GB at N=23,558).

## 8. Packaging

`scripts/ci/check_xtrax_pin.py`: replace the hard-coded `FLOOR = "0.4.0a6"` (`:14`) with the base floor parsed from pyproject by a regex that accepts `xtrax>=X,<U`
and `xtrax[extra,...]>=X,<U`; CI runs it (`ci.yml:61-63`). `tests/scripts/test_check_xtrax_pin.py` asserts the parsed floor equals the base specifier and that the `agent` extra's
floor is >= it (a test with an `xtrax[plugin]>=...` string proves the regex). `uv.lock` regenerates in the same PR.
`CHANGELOG.md` entry; README "Agent CLI/MCP" section (~10 lines). Not added: `agent_plugin.json`, skills, `.mcp.json`, PDB output (`.npz` only),
GPU in-process, a lazy `__init__`, long-run executors or checkpointing, migrating `_b1_paramize`.

## 9. Old vs new

| Concern | Today | After |
|---|---|---|
| Entry points | none | `prolix`, `prolix-mcp` |
| PDB loading | `from_pdb` (no parameters, silent) or an unpackaged script | `load_system_bundle`, fails closed |
| Bundle factory | `NameError` on every call | fixed by #46 (dependency) |
| Long runs | hand-edited `.slurm`, manual submit | `render_job` + `submit_job` (dry-run default) |
| Outputs | `outputs/` by convention | `out_dir` + JSON record (sha256s, versions, cache setting) |
| xtrax floor | pyproject `0.4.1b2`, checker `0.4.0a6` | one floor, checker reads pyproject |

## 10. Tasks (one PR; commits in order)

- **Prerequisite, not a task:** maraxen/prolix#46 (`use_size_buckets` fix) must be on `main` before Task 1; rebase onto it. Do not re-fix it here.
- **Task 1, scaffold**: `agent/{__init__,errors,_runtime}.py`, pyproject scripts/extra/floor, `check_xtrax_pin.py`, lock. Gate: `import prolix.agent`
  loads none of cisternal/fastmcp/cyclopts; pin test green. ~150 LOC.
- **Task 2, operations and loader**: `system.py`, `ops.py`, caps calibration (table goes in the PR). Gate: round-trip tests. ~400 LOC.
- **Task 3, render/submit**: `render.py`. Gate: golden test. ~200 LOC.
- **Task 4, wiring**: `tools.py`, `cli.py`, `mcp.py`, README, changelog. Gate: parity tests. ~250 LOC.
- **Task 5, tests** (§11), fixture `tests/agent/data/mini.pdb` = residues 1-3 of `data/pdb/1CRN.pdb` (`awk '$1=="ATOM" && $6<=3'` plus `END`;
  20 atoms; parameterises in the probe, A8). ~500 LOC.

## 11. Tests (`tests/agent/`; narrow local runs, CPU)

CI installs the agent extra (D9), so nothing here skips there; `importorskip("cisternal")` stays only as a convenience for local runs without the extra.
**CI (fast, default marks):** registry parity, missing-extra, refusals, render golden and stub-`uv` run, submit gating, import lightness, no-fork grep, isolation,
`format_success` stdout, concurrency/busy, exit-code table, device policy, and every op test that uses an injected tiny bundle (`bundle=` seam: a 3-atom bundle from
`tests/api/test_ensemble_plan_helpers._make_minimal_bundle`, no proxide load, small compile). **`slow` (not in CI):** the `mini.pdb` end-to-end round trip with the
real loader, the `_b1_paramize` parity test, the cap wall-clock test. Task 5 records measured wall times for both groups in the PR.
- **Registry parity:** `EXPECTED_TOOLS` (seven names) == `set(TOOL_NAMES)` == the `prolix` registry snapshot == the `prolix_mcp` registry snapshot == `build_server()`
  tool names == cyclopts command names. Control: a registry missing one name makes `wire(expected=)` raise. Each MCP twin is `inspect.iscoroutinefunction`; each CLI tool is not.
- **Missing extra:** with `cisternal` import blocked (`builtins.__import__` raising `ModuleNotFoundError(name="cisternal")`), importing `prolix.agent.cli`
  raises the `prolix[agent]` `ImportError` hint, not the raw error. Same for `mcp`.
- **Round trip, all tools** (CI: injected tiny bundle; slow: `mini.pdb`): op call and CLI call (`cli.app([...,"--json"], result_action="return_value")` with captured stdout, which
  must parse as one JSON object and contain nothing else) with a tmp `out_dir`. Assert record exists, schema, PDB sha256 equals `hashlib`,
  `jax_cache` key present, output hashes match files, no `*.tmp` left.
- **Golden vs library** (same process, same flags): `energy_breakdown.total` and `.bonded` equal the direct `energy_fn_from_bundle` /
  `bonded_energy_fn_from_bundle` values (`rtol=1e-6`); `minimize` positions equal `run_minimization(ef, x, sd_steps=n)`; `md_smoke` final positions
  `allclose` `run_nve` called with `dt_fs / AKMA_TIME_UNIT_FS` (control: calling the tool's dt as raw AKMA differs by a factor ~49 and is detected; a test of the unit conversion with a stub `run_nve` records `dt`); `minimize` asserts after <= before; `load_system_bundle` (slow) energy equals `_b1_paramize.paramize_pdb_to_bundle` (importable via `pythonpath=["."]`, pyproject). Controls:
  moving one atom 0.1 Å changes the total beyond tolerance; a `from_pdb` bundle is refused by the all-zero-parameter check.
- **Refusals** (no compile, fast): atoms over a monkeypatched cap of 5; steps over cap; `dt` over cap; missing file (CLI exit 4, cap refusal exit 3, message names
  the cap and `render_job`); existing record without `overwrite`; `script` with `..`, absolute path, non-`.py`, outside `scripts/`; `time=12:00:01`; `submit_job` on a
  name without a matching `render_job.json`; `myxcel` absent. Each asserts nothing was written.
- **Rendered-script golden:** `render_job(script="scripts/benchmarks/prolix_vs_openmm_speed.py", name, preset="gpu_h200_preempt", time, args=["--n","5; rm -rf /"])`
  (test asserts the script exists) equals `tests/agent/golden/job_gpu_h200.slurm` byte for byte; the `;` argument is single-quoted; `bash -n` exits 0.
  `submit_job` with a stub `myxcel` first on PATH (a tmp script echoing argv): `--dry-run` present by default, absent with `confirm=True`. Real `myxcel`
  and sbatch are never invoked.
- **Device policy** (subprocess): `import jax` first, then an op: backend is `cpu` and recorded; with `jax.default_backend` patched to return `"gpu"` the op raises
  `RefusalError`; `import prolix` leaves `backends_are_initialized()` false.
- **Import lightness** (subprocess): `import prolix.agent`, `.ops`, `.render` and plain `import prolix` leave `cisternal`, `fastmcp`, `cyclopts` out of
  `sys.modules`; positive control: `import prolix.agent.tools` puts all three in.
- **No-fork grep** of `src/prolix/agent`; positive control: a temp file with the token is flagged by the same helper.
- **Host-config isolation fixture** (autouse; #3527 lesson): `XDG_CONFIG_HOME` and `HOME` to tmp dirs; clear `XTRAX_JAX_CACHE*`, `JAX_COMPILATION_CACHE_DIR`,
  `PROLIX_USE_PALLAS`, `MYXCEL_*`; restore `jax_cache._applied` and JAX flags; `cisternal.init` gets `log_dir=<tmp>`. A test snapshots mtimes of `~/.cisternal`,
  `~/.config/xtrax`, `~/.myxcel.toml` before and after and asserts none was **written**.
- **More controls:** an H-free PDB yields the warning and `n_hydrogen: 0`; `out_dir` through a symlink to outside the root is refused; killing the writer between side file and record (monkeypatched `os.replace` failure) leaves no record; a refused call leaves `out_dir` empty; `neighbor` is never passed to the energy function (spy); the bundle cache returns `hit` on a second call with the same sha256 and `miss` after the file changes.
- **Budget:** CI group target under 90 s total, measured and recorded; anything over moves to `slow`. A `slow` test runs the largest allowed combination under 120 s.

## 12. Assumption ledger

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | the next xtrax release after 0.4.1b9 ships `xtrax.agent` and `xtrax[plugin]` | pin and lock cannot be written | UNVERIFIED | deferred: owner's release process; xtrax PR #284 is unreleased |
| A2 | `from_pdb` applies no force-field parameters | loader choice changes | VERIFIED | read: src/prolix/types/bundles.py:277-297,336-345 |
| A3 | `make_bundle_from_system` raised `NameError` at `use_size_buckets`, now fixed by prolix#46; a second `NameError` at `_chunked_lj_energy_nl_core` is debt #3586 on the NL path only | the tools would break on bundle build or on the NL path | UNVERIFIED | deferred: #46 not merged in this base; read: src/prolix/physics/system.py:874-876 (pre-fix), src/prolix/physics/optimization.py:363 (NL fwd rule); the tools never pass `neighbor`, enforced by a test |
| A4 | `run_minimization` uses only `sd_steps`; `run_nve`/`run_nvt` use key 0 and return positions only | caps and record fields change | VERIFIED | read: src/prolix/physics/simulate.py:141-197,259-328 |
| A5 | `scripts/` is not packaged | the loader could be imported from scripts | VERIFIED | read: pyproject.toml:50-51 |
| A6 | `jax.config.update("jax_platforms","cpu")` after `import jax` and before device use selects CPU; xtrax `jax_cache` initialises no backend | CPU policy needs an env var set before import | UNVERIFIED | deferred: spike-run tool absent here (scripts/loop/adversarial_metrics.py not in this workspace); an inline probe printed `[CpuDevice(id=0)]` on jax 0.10.2 from the main checkout's older venv, not the worktree lock; read: xtrax src/xtrax/jax_cache.py (no `devices(`/`default_backend` call) |
| A7 | `jax_cache.current()` returns the applied setting; `as_record()` is JSON-ready | record key changes | VERIFIED | read: xtrax src/xtrax/jax_cache.py:223-240,827-836 |
| A8 | `mini.pdb` (20 atoms) and 1CRN (327 atoms) parameterise through `_b1_paramize` once #46 is applied, with finite energies and the timings used for calibration | fixture and caps change | UNVERIFIED | deferred: spike-run tool absent; inline probe on a patched temp copy (CPU, 4 threads, older venv): 20 atoms load 13.5 s, grad 13.2 s, 50 sd 2.4 s, 200 NVE 2.7 s; 327 atoms 9.3 s, 14.7 s, 3.1 s, 3.7 s |
| A9 | `import prolix` takes ~13 s and imports none of cisternal, fastmcp, cyclopts | lightness test and `--help` note change | UNVERIFIED | deferred: spike-run tool absent; one cold inline run, 12.9 s, three `in sys.modules` checks False |
| A10 | cisternal runs sync tool bodies on the event loop, so FastMCP worker threads cannot be relied on (replaces the earlier wrong thread assumption) | async twins unnecessary | VERIFIED | read: cisternal src/cisternal/registration/compose.py:130-133; shim.py:62-64 |
| A11 | `prolix.__version__` is "0.1.0" while pyproject says 1.0.0, so the record reads `importlib.metadata` | record shows a wrong version | VERIFIED | read: src/prolix/__init__.py:3; pyproject.toml:7 |
| A12 | `myxcel submit REMOTE PROJECT SCRIPT_PATH` accepts `--dry-run`/`--respect-directives`, requires a local script, and maps in-tree paths to the remote | contract changes | VERIFIED | read: myxcel src/myxcel/cli.py:1689,1792-1830 |
| A13 | the project name `prolix` and remote `engaging` exist in the owner's myxcel config | `submit_argv` defaults change | UNVERIFIED | deferred: remote profiles live in the owner's `~/.config/myxcel`, not read on purpose; both are parameters |
| A14 | `bonded_energy_fn_from_bundle` is the bonded-only energy | breakdown drops to total only | VERIFIED | read: src/prolix/api/bundle_md.py:223,579-604 |
| A16 | cisternal's CLI path calls tool bodies synchronously, so async tools need a separate MCP registry | one async tool would return an unawaited coroutine on the CLI | VERIFIED | read: cisternal src/cisternal/registration/wired.py:103 (`apply_recovery_sync`) |
| A17 | dt units are AKMA (1 unit = 48.888 fs) in `run_nve`/`run_nvt_nose_hoover`; cyclopts exits 2 on usage errors | dt conversion or exit table changes | UNVERIFIED | read: src/prolix/simulate.py:56 and physics/simulate.py:283,310 settle the units; the cyclopts exit code is deferred to a Task 4 test |
| A18 | CI runs `uv sync --extra dev` and `uv run ty check` | CI changes unneeded | VERIFIED | read: .github/workflows/ci.yml:36,59 |
| A19 | `format_success` is required for `--json` output; `json_option` only adds a flag | `--json` would be inert | VERIFIED | read: cisternal src/cisternal/registration/cli_contract.py:118-133 |
| A15 | cisternal: `@tool(registry=...)`, `wire(server, app, registry, expected, cli_contract)`, no middleware from `wire`, fork unsupported; `src/prolix` uses no `multiprocessing` | wiring or spawn policy changes | VERIFIED | read: cisternal src/cisternal/registration/decorator.py:54-65; wired.py:408-420; src/prolix/tiling/planner.py:9 is the only `fork` hit (a comment) |

## 13. Risks

| Risk | Mitigation |
|---|---|
| Debt #3586 (second NameError, NL path) | tools stay on the dense path; test asserts no `neighbor` argument |
| Caps wrong on other machines | lower-only constants, calibration table in the PR, slow wall-clock test |
| `submit_job` reachable by an agent | dry-run default, `confirm`, only rendered scripts with matching sha256, argv list without a shell |
| Rendered script drifts from `scripts/slurm` conventions | golden file reviewed against `b1_init_exec.slurm`; presets in one table |
| Loader copy drifts from the script | parity test |
| `prolix --help` pays the 13 s import | documented; lazy `__init__` is the follow-up |
| xtrax release slips | PR stays draft; rollback is reverting it (no library code changes) |

## 14. Delivery

One PR to prolix `main` off 5ce098d, opened after the xtrax release (A1) and depending on prolix#46. Commit order: Tasks 1-5. Gates are narrow
local runs (`pytest tests/agent -q`, `tests/physics/test_bundle_factory.py`); the full suite runs in CI only. The PR body lists the calibration table
and the pin value. Reviewer reads first: `ops.py` caps, `render.py` quoting and presets, `submit_job` guards. Follow-up debts to file: lazy
`prolix.__init__`; stale `scripts/slurm/install_xtrax_wheel.slurm:99-107` assertion (`0.4.0a7`); `from_pdb` docstring versus behaviour.

## 15. Review log

Round 1 (challenger C1-C17), verified at source; all accepted except as noted.
- C1 accepted: cisternal sync bodies run on the event loop; async `to_thread` twins in a second registry, one-op lock with immediate `busy`, tested (D7, §4).
- C2 accepted: `dt_fs` converted by `AKMA_TIME_UNIT_FS`; unit test. C3 accepted: honest latency, per-process bundle/closure cache keyed by sha256, `bundle=` seam.
- C4 accepted: CI installs `--extra agent` (also for `ty`); CI vs slow test split listed (D9, §11). C5 accepted: `format_success` defined, stdout tested.
- C6 accepted: absolute artifact path, `cwd=repo_root`, stub-myxcel mapping test. C7 accepted: threat model, env-var gate `PROLIX_AGENT_ALLOW_SUBMIT`, sha256 pinning, script allowlist, no re-render (D8).
- C8 accepted: template sources `_uv_env.sh` like `l3_smoke.slurm`, header provenance corrected, `rc` capture, stub-`uv` failure test. C9 accepted: conftest `ensure_cpu`.
- C10 accepted: `periodic` dropped. C11 modified: H-free PDBs warn and record `n_hydrogen` rather than refuse, because the probes show the loader adds no hydrogens and the fixtures are H-free. C12 recorded in A6/A8/A9 (old venv, jax 0.10.2). C13 accepted: out_dir root, `.tmp.npz`.
- C14 accepted: minimize assertion, `bonded` defined (exact term list confirmed from a read in Task 2). C15 accepted: exit-code table. C16 accepted: base floor unchanged, regex handles extras.
- C17 accepted: Task 0 dropped, depends on #46; debt #3586 scoped to the NL path, tools never pass `neighbor`.
