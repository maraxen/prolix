---
title: 'prolix CLI and MCP server: quick in-process operations, long runs rendered as jobs'
description: 'Implementation spec for debt #3555: a cisternal-wired prolix CLI and MCP server (src/prolix/agent) exposing seconds-scale operations in-process on CPU and rendering or submitting long runs through myxcel; built on xtrax.agent.'
status: converged
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
| D3 | `render_job` wraps a repo script in an sbatch file under a path `myxcel push` ships and returns the exact `myxcel push` + `myxcel submit` argvs; `submit_job` runs them, dry-run unless `confirm=true`; non-preemptible presets only | a prolix long-run executor (no checkpoint API); submit by default; `myxcel submit-job PRESET COMMAND` (no reviewable script) |
| D4 | `out_dir` is a required argument for in-process tools; `render_job` writes to `scripts/slurm/agent_jobs/` inside the checkout (git-not-ignored, so `myxcel push` ships it; §6) | default `outputs/` or cwd |
| D5 | Copy the b1 loader into the package, with a parity test against the script | import from `scripts/`; refactor the script to call the package |
| D6 | Base `xtrax>=0.4.1b2,<0.5` unchanged; only the `agent` extra pins `xtrax[plugin]>=X,<0.5`; `check_xtrax_pin.py` reads the base floor from pyproject (regex tolerant of extras) | raising the base floor for every library user; two hard-coded floors |
| D7 | MCP tool wrappers are `async` and run the op via `asyncio.to_thread` under an `asyncio.Lock`-guarded semaphore of 1; a second concurrent call refuses immediately (`busy`) | sync bodies (cisternal runs them on the event loop, A10); queueing silently |
| D8 | `submit_job` is disabled unless `PROLIX_AGENT_ALLOW_SUBMIT=1` is in the server/CLI environment (set by the human, out of band) | `confirm` flag as the only gate (an agent can set it) |
| D10 | Console scripts point at `prolix_agent_entry` (a top-level package outside `prolix`) which sets `JAX_PLATFORMS=cpu` before `prolix` is imported (§7) | `jax.config.update` inside `main()` (too late: `import prolix` initialises the backend); refactoring `prolix/__init__` or `cmap.py` (out of scope, follow-up) |
| D9 | CI installs `--extra dev --extra agent`; ops take an optional pre-built `bundle=` seam so tests skip the 13 s load | importorskip everywhere (everything skips in CI) |

## 3. Layout and entry points

```
src/prolix_agent_entry/__init__.py  prepare(), cli_main(), mcp_main(): top-level package, imports nothing from prolix at import (§7)
src/prolix/agent/__init__.py   require_agent_extra() raising AgentExtraMissing(ImportError) (aminx/agent/__init__.py pattern); imports nothing heavy
src/prolix/agent/errors.py     AgentError > InputError, RefusalError
src/prolix/agent/_runtime.py   assert_cpu(), cap constants, run lock, versions(), sha256(), atomic_write(), write_record()
src/prolix/agent/system.py     load_system_bundle(path, ff_name, periodic): copy of _b1_paramize.paramize_pdb_to_bundle
src/prolix/agent/ops.py        pure functions, one per operation (no cisternal import); golden tests call these
src/prolix/agent/render.py     render_job_script(), submit_argvs(), load_allowlist() (no cisternal import)
src/prolix/agent/job_scripts.txt  render allowlist (§6; package data)
src/prolix/agent/tools.py      TOOL_NAMES; sync tools in registry "prolix" (CLI) and async to_thread twins in "prolix_mcp" (MCP), same names
src/prolix/agent/cli.py        require_agent_extra() on its FIRST line; cyclopts app; wire(None, app, ...); main() (calls xtrax.agent.run_cli(app, app_kwargs={"exit_on_error": False, "print_error": True}) inside a wrapper mapping cyclopts.CycloptsError -> SystemExit(2), the demistify convention, demistify cli_support.py:226-257)
src/prolix/agent/mcp.py        require_agent_extra() on its first line; build_server(); main() (calls xtrax.agent.serve)
```

`pyproject.toml`: `[project.scripts] prolix = "prolix_agent_entry:cli_main"`, `prolix-mcp = "prolix_agent_entry:mcp_main"` (the entry package sits under `src/`, so `packages.find where=["src"]` ships it; A20);
`[tool.setuptools.package-data] "prolix.agent" = ["job_scripts.txt"]`;
`agent = ["xtrax[plugin]>=X,<0.5"]` (cisternal, cyclopts, fastmcp come transitively); base `xtrax>=0.4.1b2,<0.5` is unchanged. `X` = first
xtrax release containing `xtrax.agent` (after 0.4.1b9; A1). `cli.py` and `mcp.py` call `require_agent_extra()` before any cisternal,
cyclopts or fastmcp import (they wire at import time), so a missing extra prints the `prolix[agent]` hint instead of a
`ModuleNotFoundError`. CI (`ci.yml:36,59`) today runs `uv sync --extra dev` and `uv run ty check`: Task 1 changes both steps to `--extra dev --extra agent`, so the cisternal imports type-check and agent tests run rather than skip (C4). `prolix/__init__.py` does not import `prolix.agent`; `ops/render/_runtime` import none of the three. Heavy imports
(`jax`, `proxide`, `prolix.api`) sit inside operation bodies. `import prolix` costs ~13 s cold (A9), so `prolix --help` pays it;
a lazy `prolix/__init__` is a follow-up.

```python
# src/prolix_agent_entry/__init__.py: no prolix import at module level
def prepare() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"          # unconditional; replaces any inherited value, BEFORE `import prolix`
def cli_main() -> None:
    prepare()
    _require_agent_extra()                       # find_spec checks for cisternal/fastmcp/cyclopts/xtrax.agent; exit 1 + hint, BEFORE any prolix import (~13 s)
    from prolix.agent import cli
    cli.main()                                   # -> xtrax.agent.run_cli(app)
def _require_agent_extra() -> None: ...         # importlib.util.find_spec only (imports nothing heavy); on a missing module print the `prolix[agent]` hint to stderr, raise SystemExit(1)
def mcp_main() -> None: ...                      # same shape, prolix.agent.mcp.main() -> xtrax.agent.serve(build_server())
```

`prolix/__init__.py` initialises the JAX backend at import (`physics/cmap.py:39` builds a `jnp.array`; `prolix.physics.system` imports it, A6), so any `prolix.agent.*`
module is imported after the backend exists and a `jax.config.update` in `main()` is too late. The env var must be set before the first `import prolix`;
hence the entry package outside `prolix`. `cli.main()`/`mcp.main()` therefore contain no CPU logic; they exist so `python -c` callers and tests can reach the app.
`python -m prolix.agent.cli` is not a supported entry (no `__main__`); a caller that imports `prolix` first is caught by `assert_cpu()` (§7). Follow-up, not this spec: make `prolix/__init__`
or `cmap.WT` lazy.

`build_server()` mirrors `aminx.agent.mcp.build_server`: `FastMCP("prolix", instructions=...)`, `cisternal.wire(server, registry="prolix_mcp",
expected=list(TOOL_NAMES))` (the async twins, D7: cisternal's CLI path is sync, `wired.py:103`, so one async function cannot serve both), `add_middleware(CisternalMiddleware(adapter=PassthroughAdapter(), reraise=True))`. `cli.py`:
`wire(None, app, registry="prolix", expected=TOOL_NAMES, cli_contract=CONTRACT)` at import, so a renamed or dropped tool fails at
import (demistify `cli.py:1013`). `CONTRACT`: `exit_codes={InputError: 4, RefusalError: 3}`, `options=[json_option()]` and a `format_success(result, ctx)` that writes
`json.dumps(result, sort_keys=True)` to stdout when `ctx.options["json_out"]` else `key: value` lines; `json_option` alone changes nothing (C5). Nothing else may
write to stdout; logs go to stderr. Tool bodies return dicts and
raise `AgentError`; they never print (cisternal rule 2).

**Exit codes (published, tested; each row has its own test):**

| Code | Meaning | Source | Test |
|---|---|---|---|
| 0 | success | | any round trip |
| 1 | `xtrax[plugin]` or `prolix[agent]` missing | `prolix_agent_entry._require_agent_extra()` checks the extra before any `prolix` import and exits 1 with the hint (`require_agent_extra`'s `AgentExtraMissing` stays the guard for direct `prolix.agent` importers); `xtrax.agent` also exits 1 for missing cisternal (same meaning) (`xtrax/agent.py:98`) | cisternal import blocked, subprocess of `cli_main`: exit 1, stderr names `prolix[agent]` |
| 1 (overlap) | unhandled exception (Python's traceback status) | interpreter | documented, not a contract; stderr carries a traceback, not the hint |
| 2 | cyclopts usage error (bad type, missing argument, unknown command), mapped to 2 by the `CycloptsError` wrapper so the code does not depend on the cyclopts version (A28) | `prolix.agent.cli.main` wrapper | `prolix load_system --bogus` exits 2 |
| 2 (overlap) | bad jax-cache configuration | `xtrax.agent` (`agent.py:93`), stderr prefix `prolix: jax cache configuration error:` | malformed cache setting, trigger copied from xtrax's own test for this exit; assert exit 2 **and** the stderr prefix |
| 3 | `RefusalError` (caps, busy, submit disabled) | `CONTRACT.exit_codes` | cap refusal |
| 4 | `InputError` | `CONTRACT.exit_codes` | missing file |

Codes 2 are not distinct: both are "retrying with the same arguments cannot help", and the stderr prefix tells them apart; the overlap is printed in the `--help` epilog. The table lives in the epilog.

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
| `md_smoke` | short NVE or NVT Nose-Hoover; arguments `steps`, `dt_fs`, `ensemble` (`nve`\|`nvt`), `temperature_k` (NVT only; passing it with `nve` is `InputError`, because `run_nve` hard-codes 300 K initial velocities and key 0 at `physics/simulate.py:284-286`) | `run_nve(ef, x0, steps, dt, mass=masses)` / `run_nvt_nose_hoover(ef, x0, steps, dt, temperature=temperature_k, mass=masses)`, `masses = bundle.masses[:n_real]` (the default `mass=1.0` would be unit masses), `dt = dt_fs / AKMA_TIME_UNIT_FS` (`prolix/simulate.py:56`; AKMA units, `physics/simulate.py:283,310`) | `final.npz` | `finite`, start/end energy, steps, dt, ensemble, `temperature_k` (NVT) or `init_velocity_temperature_k: 300 (fixed by run_nve)` (NVE), `seed:"fixed 0"`; no conservation claim |
| `render_job` | write an sbatch script (§6) | `render.render_job_script` | `scripts/slurm/agent_jobs/<name>/job.slurm` | path, sha256, resources, `submit_argvs` |
| `submit_job` | `myxcel push` then `myxcel submit` (§6) | `render.submit_argvs` + `subprocess.run` | none | `dry_run`, per-step argv/exit/stdout/stderr, `slurm_id`, `follow` |

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

**Concurrency (D7, A10; the CLI path is sync, A16):** cisternal runs a sync tool body directly on the event loop (compose.py:130-133 via `shim.dispatch`, shim.py:62-64), so a 25 s
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

Today (read): `scripts/slurm/*.slurm` carry `#SBATCH` headers, `source scripts/slurm/_common_env.sh` (cd to repo root, lock-keyed uv venv via `_uv_env.sh`
under myxcel), then `uv run python scripts/...`. `myxcel.toml` has `[uv] venv = "lock-keyed"`. **`myxcel submit REMOTE PROJECT SCRIPT_PATH` does not upload anything**:
it requires the script to exist locally and translates a path inside the project's local tree to the remote path, so the file must already be on the remote
(`cli.py:1792-1830`, A12). **`myxcel push` ships exactly `git ls-files --cached --others --exclude-standard` minus profile excludes** (`rsync.py:173-182,69-80`), and
`.gitignore:27` ignores `outputs/`, so an artifact under `outputs/` would never reach the remote and the submit would fail there (A21). Hence the artifact lives under
`scripts/slurm/agent_jobs/`, which no ignore rule matches (`git check-ignore` exit 1, A22); untracked-but-not-ignored files are pushed. No `.gitignore` entry may be added for that
directory (it would silently break the push). Rendered dirs stay untracked; a human deletes them.
No prolix executor exists for production MD, ensembles, REMD or fitting: they are ad hoc scripts with no checkpoint or resume API (§1.4).

**Contract.** `render_job(script, args, preset, time, name, repo_root=None)`:
- `script`: repo-relative path under `scripts/`, `.py`, existing, resolving inside `repo_root` (`..`, absolute, symlink escapes refused), **and listed in
  `src/prolix/agent/job_scripts.txt`**. `repo_root` defaults to the nearest ancestor of cwd with `pyproject.toml` `name = "prolix"` and `scripts/slurm/_uv_env.sh`, else `InputError`.
- **`job_scripts.txt` format (exact):** UTF-8, one repo-relative POSIX path per line; blank lines and lines whose first non-space character is `#` are ignored; no globs; no other syntax.
  Initial content, exactly one path line: `scripts/benchmarks/prolix_vs_openmm_speed.py` (exists, argparse CLI with `--sweep`/`--warmup`, A23); the PR author adds more only by naming them in the PR.
  Read only through `render.load_allowlist()` using `importlib.resources.files("prolix.agent").joinpath("job_scripts.txt")` (shipped as package data, §3). An empty or missing-entry list renders nothing (`InputError`).
- Artifact: `<repo_root>/scripts/slurm/agent_jobs/<name>/job.slurm` and `render_job.json` (script sha256, source script sha256, args, preset, time, git rev).
  Re-rendering an existing `<name>` refuses (`InputError`), so a rendered script is immutable after review. The one exception to D4.
- `args`: strings, each `shlex.quote`d. `preset` closed table (**non-preemptible only**):
  `cpu_quick` (`#SBATCH --partition=mit_quicktest --time=0:10:00 --cpus-per-task=2 --mem=4G`, headers of `l3_smoke.slurm:2-9`) or
  `gpu_l40s` (`--partition=mit_normal_gpu --gres=gpu:l40s:1 --cpus-per-task=16 --mem=64G`, headers of `prof_ensemble_apply_probe.slurm:2-7`).
  `time` `HH:MM:SS`; defaults and maxima: `cpu_quick` 0:10:00, `gpu_l40s` 2:00:00 (below the 5:45 `mit_normal_gpu` cap, A24). `name` `[A-Za-z0-9_-]{1,40}`.
  **Why no `mit_preemptable` preset:** a preempted job is requeued and restarts from step 0, because the target scripts have no checkpoint API (§1.4). A preemptable preset is added only together with
  a chunked, resumable runner (each unit its own process and timeout, completion stamp and sha256 of input and artifact, reuse only on a hash match); that is the follow-up, and `--auto-requeue` is rejected under
  `--respect-directives` anyway (`cli.py:1761`, A25). The 2 h cap bounds one job; there is no multi-hour single timeout.
- Template (golden-tested; header sources `l3_smoke.slurm:34-35` and `b1_init_exec.slurm:111-115`, not `_common_env.sh`, whose `write_engaging_manifest.py` call is a side effect we do not want):
  ```
  #!/bin/bash
  <preset #SBATCH lines> --job-name=prolix_<name> --output/--error=scripts/slurm/agent_jobs/<name>/%j.<name>.{out,err}
  set -euo pipefail
  cd "${SLURM_SUBMIT_DIR:?}"
  mkdir -p scripts/slurm/agent_jobs/<name>   # no-op safeguard; Slurm opens the log before this runs, so the dir must already exist remotely (it does: the script is pushed into it, A31)
  source scripts/slurm/_uv_env.sh && prolix_uv_sync
  rc=0
  uv run python <script> <quoted args> || rc=$?
  echo "=== <name> exit=${rc} ==="
  exit "${rc}"
  ```
  `|| rc=$?` keeps the status line reachable under `set -e` (the golden test also runs the template with a stub `uv` that fails and asserts the line prints and the script exits non-zero). No `bth run`.
- Result: `artifact` (absolute path), `submit_argvs` (the two argvs below), `notes`.

`submit_argvs = [push, submit]`, both with `cwd=repo_root`:
- push: `["myxcel","push",<remote>,"prolix","--apply","--yes"]` (`--dry-run` instead of `--apply --yes` unless `confirm`; push is dry-run by default and `--apply` performs it, `cli.py:433-445`).
- submit: `["myxcel","submit",<remote>,"prolix",<absolute artifact path>,"--respect-directives"]` (`--dry-run` appended unless `confirm`).
myxcel maps the script path only under `profile.local_workspace/<project>` or the detected worktree root (`cli.py:1793-1820`); elsewhere it prints "script is outside ..." and the remote path must already exist, so `notes` states `repo_root` must equal the myxcel project root.
`push` is additive and ships the whole non-ignored tree, not just the script; `notes` says so.

`submit_job(name, remote="engaging", confirm=false, repo_root=None)`:
- Threat model: the caller is an LLM agent that can set any tool argument and read tool output, but cannot set the server's or CLI's process environment. So `confirm`
  is a typo guard only; the authorisation is **`PROLIX_AGENT_ALLOW_SUBMIT=1` in the process environment** (D8). Unset: any non-dry-run attempt raises `RefusalError`; dry-run still works.
  Residual risk: an agent that can edit the human's shell profile or launch its own server; out of scope, named in the PR.
- Refuses unless `render_job.json` exists, the artifact's sha256 equals the recorded one, and the recorded source-script sha256 still matches the file in `repo_root`. Runs push, then (only if push exit 0) submit, each
  `subprocess.run(argv, shell=False, timeout=120, capture_output=True, cwd=repo_root, env=os.environ minus JAX_PLATFORMS)`. Returns per-step argv, exit code, stdout, stderr, plus `slurm_id`:
  parsed from the submit stdout line matching `^slurm_id:\s*(\d+)$` (myxcel prints it at `cli.py:2175`, A26), `null` on dry-run or no match (never guessed).
  **Following the job:** the result's `follow` field names myxcel's `job_status(remote, slurm_id)` MCP tool (`myxcel/tools.py:116`); prolix adds no status tool (the owner's myxcel MCP server already has one, A26).
  Missing `myxcel` raises `InputError` with both argvs for a human to run. `subprocess` is spawn-safe; cisternal forbids only `fork` multiprocessing.
- **`--dry-run` proves:** the artifact and its sha256 pins check out, `myxcel` resolves the profile/project and maps the script path, and the sbatch line myxcel would run is printed. **It does not prove:**
  that the remote has the file (push dry-run only lists what would transfer), that the partition/gres/time are accepted by Slurm, that the uv env builds, or that the script runs.

## 7. Process, device and multiprocessing policy

- Startup is `xtrax.agent` only; `main()` never calls `jax_cache.configure()` or `cisternal.init()` itself.
- **CPU policy (D10):** `prolix_agent_entry.prepare()` sets `os.environ["JAX_PLATFORMS"]="cpu"` (unconditionally, overriding an inherited value) before `prolix` is imported. Probe: with the
  variable set first, `import prolix` then yields backends `['cpu']` and `default_backend()=='cpu'` (A6); without it the backend is initialised at import by `cmap.py:39` (A6), so later
  `jax.config.update` cannot help. The earlier "`ensure_cpu()` via `jax.config.update`" is removed.
- `_runtime.assert_cpu()` (each in-process op calls it first): raises `RefusalError` unless `jax.default_backend() == "cpu"`. Fails closed for callers who import `prolix` before the entry point (a library user, `python -m`).
  `xtrax/jax_cache.py` has no `devices(`/`default_backend`/`xla_bridge` call (A6). Pallas stays default-off (`pallas_kernels.py:81-92`, enabled only by `PROLIX_USE_PALLAS`); the record stores the flag. x64 is untouched
  (tests flip it per fixture, `tests/physics/conftest.py:66-80`). No `device=` argument is offered.
- Tests: `tests/agent/conftest.py` cannot set the env early enough inside a pytest process that already imported jax (other conftests import it), so it asserts `jax.default_backend()=="cpu"` at session start and calls
  `pytest.exit` naming `JAX_PLATFORMS=cpu pytest tests/agent` otherwise (loud, not silent). The real-entry-point test is a subprocess (§11 Device policy).
- No `multiprocessing`: `src/` has none (A15); a test greps `src/prolix/agent` for `multiprocessing`/`os.fork`.
- `MAX_ATOMS` bounds the dense `N x N` exclusion matrices (`bundle_md.py` docstring: ~4.4 GB at N=23,558).

## 8. Packaging

`scripts/ci/check_xtrax_pin.py`: replace the hard-coded `FLOOR = "0.4.0a6"` (`:14`) with the base floor parsed from pyproject by a regex that accepts `xtrax>=X,<U`
and `xtrax[extra,...]>=X,<U`; CI runs it (`ci.yml:61-63`). `tests/scripts/test_check_xtrax_pin.py` asserts the parsed floor equals the base specifier and that the `agent` extra's
floor is >= it (a test with an `xtrax[plugin]>=...` string proves the regex). `uv.lock` regenerates in the same PR.
`CHANGELOG.md` entry; README "Agent CLI/MCP" section (~10 lines). Not added: `agent_plugin.json`, skills, `.mcp.json`, PDB output (`.npz` only),
GPU in-process, a preemptable job preset (needs a resumable runner), a prolix job-status tool (use myxcel's `job_status`), a lazy `__init__`, long-run executors or checkpointing, migrating `_b1_paramize`.

## 9. Old vs new

| Concern | Today | After |
|---|---|---|
| Entry points | none | `prolix`, `prolix-mcp` |
| PDB loading | `from_pdb` (no parameters, silent) or an unpackaged script | `load_system_bundle`, fails closed |
| Bundle factory | `NameError` on every call | fixed by #46 (dependency) |
| Long runs | hand-edited `.slurm`, manual submit | `render_job` + `submit_job` (push then submit, dry-run default, non-preemptible presets, `slurm_id` returned) |
| Outputs | `outputs/` by convention | `out_dir` + JSON record (sha256s, versions, cache setting) |
| xtrax floor | pyproject `0.4.1b2`, checker `0.4.0a6` | one floor, checker reads pyproject |

## 10. Tasks (one PR; commits in order)

- **Prerequisite, not a task:** maraxen/prolix#46 (`use_size_buckets` fix) must be on `main` before Task 1; rebase onto it. Do not re-fix it here.
- **Task 1, scaffold**: `prolix_agent_entry/__init__.py`, `agent/{__init__,errors,_runtime}.py`, `agent/job_scripts.txt`, pyproject scripts/extra/package-data/floor, `check_xtrax_pin.py`, lock. Gate: `import prolix.agent`
  loads none of cisternal/fastmcp/cyclopts; `import prolix_agent_entry` leaves `prolix` unimported; pin test green; `uv build --wheel` output contains `prolix_agent_entry/` and `prolix/agent/job_scripts.txt` (`unzip -l`). ~170 LOC.
- **Task 2, operations and loader**: `system.py`, `ops.py`, caps calibration (table goes in the PR). Gate: round-trip tests. ~400 LOC.
- **Task 3, render/submit**: `render.py` (allowlist loader, render, push+submit argvs, `slurm_id` parse). Gate: golden, allowlist, pushed-set and stub-myxcel tests. ~250 LOC.
- **Task 4, wiring**: `tools.py`, `cli.py`, `mcp.py`, README, changelog. Gate: parity tests, exit-code table tests, entry-point subprocess test (§11). ~250 LOC.
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
  raises `AgentExtraMissing` carrying the `prolix[agent]` hint, not the raw error; a subprocess of `cli_main` exits 1 with that hint on stderr. Same for `mcp_main`.
- **Exit-code table:** one test per row of the §3 table (rows 0, 1, 2 usage, 2 cache with stderr prefix, 3, 4), each asserting code and stderr text; the usage-error test is the guard on cyclopts' exit code in the locked version.
- **Round trip, all tools** (CI: injected tiny bundle; slow: `mini.pdb`): op call and CLI call (`cli.app([...,"--json"], result_action="return_value")` with captured stdout, which
  must parse as one JSON object and contain nothing else) with a tmp `out_dir`. Assert record exists, schema, PDB sha256 equals `hashlib`,
  `jax_cache` key present, output hashes match files, no `*.tmp` left.
- **Golden vs library** (same process, same flags): `energy_breakdown.total` and `.bonded` equal the direct `energy_fn_from_bundle` /
  `bonded_energy_fn_from_bundle` values (`rtol=1e-6`); `minimize` positions equal `run_minimization(ef, x, sd_steps=n)`; `md_smoke` final positions
  `allclose` `run_nve(..., mass=bundle.masses[:n_real])` called with `dt_fs / AKMA_TIME_UNIT_FS` (a stub `run_nve` also records that `mass` was the bundle's, not the default 1.0; `md_smoke(ensemble="nve", temperature_k=...)` is `InputError`; NVT forwards `temperature_k`) (control: calling the tool's dt as raw AKMA differs by a factor ~49 and is detected; a test of the unit conversion with a stub `run_nve` records `dt`); `minimize` asserts after <= before; `load_system_bundle` (slow) energy equals `_b1_paramize.paramize_pdb_to_bundle` (importable via `pythonpath=["."]`, pyproject). Controls:
  moving one atom 0.1 Å changes the total beyond tolerance; a `from_pdb` bundle is refused by the all-zero-parameter check.
- **Refusals** (no compile, fast): atoms over a monkeypatched cap of 5; steps over cap; `dt` over cap; missing file (CLI exit 4, cap refusal exit 3, message names
  the cap and `render_job`); existing record without `overwrite`; `script` with `..`, absolute path, non-`.py`, outside `scripts/`; `time=12:00:01`; `submit_job` on a
  name without a matching `render_job.json`; `myxcel` absent. Each asserts nothing was written.
- **Rendered-script golden:** `render_job(script="scripts/benchmarks/prolix_vs_openmm_speed.py", name, preset="gpu_l40s", time, args=["--sweep","1000; rm -rf /"])`
  (test asserts the script exists) equals `tests/agent/golden/job_gpu_l40s.slurm` byte for byte; the `;` argument is single-quoted; `bash -n` exits 0. The golden's `--output`/`--error` paths start with the artifact's own directory (`scripts/slurm/agent_jobs/<name>/`), asserted explicitly (A31).
  `args=["--sweep","1000; rm -rf /"]` (the script's real flag, A23). **Allowlist:** `load_allowlist()` returns exactly `["scripts/benchmarks/prolix_vs_openmm_speed.py"]`; every returned path exists in the checkout, starts `scripts/`, ends `.py`;
  comments/blank lines are skipped (fixture text); a monkeypatched empty list makes `render_job` raise `InputError`; the file is reachable through `importlib.resources` (and Task 1's gate checks the built wheel contains it).
  **Pushed-set (non-vacuous):** in a tmp git repo with the repo's real `.gitignore` copied, render, then run `git ls-files -z --cached --others --exclude-standard` (the command myxcel's push uses, `rsync.py:182`) and assert
  the artifact's repo-relative path is in the output; control: the same assertion for a file under `outputs/agent_jobs/` fails (proves `outputs/` would have been dropped, so the test can go red).
  `submit_job` with a stub `myxcel` first on PATH (a tmp script echoing argv): the stub receives push then submit, both with `--dry-run` by default; with `confirm=True` and `PROLIX_AGENT_ALLOW_SUBMIT=1` push has `--apply --yes` and submit has no `--dry-run`;
  a non-zero push exit stops before submit; stub stdout `slurm_id: 4242` yields `slurm_id: 4242`, any other text yields `null`. Real `myxcel` and sbatch are never invoked.
- **Device policy.** (a) *Real entry point, subprocess, poisoned env:* parent env `JAX_PLATFORMS=cuda`; the child registers an `atexit` printer of `{backends_are_initialized, list(backends()), default_backend()}`,
  sets `sys.argv=["prolix","--help"]`, calls `prolix_agent_entry.cli_main()`; assert exit 0 and the printed record is `['cpu']`/`cpu`. This fails if the env were set too late: `import prolix` would initialise
  the poisoned `cuda` backend and raise or yield a non-cpu backend. (b) *Control, same child but `from prolix.agent.cli import main` (bypasses the entry package):* asserts it does **not** end
  `['cpu']`-only with exit 0 (nonzero exit, or a backend other than cpu); if (b) passes green the test is vacuous and fails. (c) `import prolix_agent_entry` leaves `prolix` out of `sys.modules`; `prepare()` overrides a preset
  `JAX_PLATFORMS=gpu`. (d) in-process: with `jax.default_backend` patched to `"gpu"` an op raises `RefusalError`. (e) slow: `prolix load_system mini.pdb --out-dir tmp --json` with the poisoned env writes a record with `jax.backend == "cpu"`.
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
| A6 | `jax.config.update("jax_platforms","cpu")` inside `main()` selects CPU (round 1 wording) | CPU policy needs an env var set before `import prolix` | REFUTED | deferred: probe, not spike-run (spike tool absent from this workspace): after `import prolix` `prolix.physics.cmap` is in `sys.modules` and `backends_are_initialized()` is True; read: src/prolix/physics/cmap.py:39 (`jnp.array` at import); changed: §3/§7 CPU policy is `JAX_PLATFORMS=cpu` set by `prolix_agent_entry` before import (D10) |
| A7 | `jax_cache.current()` returns the applied setting; `as_record()` is JSON-ready | record key changes | VERIFIED | read: xtrax src/xtrax/jax_cache.py:223-240,827-836 |
| A8 | `mini.pdb` (20 atoms) and 1CRN (327 atoms) parameterise through `_b1_paramize` once #46 is applied, with finite energies and the timings used for calibration | fixture and caps change | UNVERIFIED | deferred: spike-run tool absent; inline probe on a patched temp copy (CPU, 4 threads, older venv): 20 atoms load 13.5 s, grad 13.2 s, 50 sd 2.4 s, 200 NVE 2.7 s; 327 atoms 9.3 s, 14.7 s, 3.1 s, 3.7 s |
| A9 | `import prolix` takes ~13 s and imports none of cisternal, fastmcp, cyclopts | lightness test and `--help` note change | UNVERIFIED | deferred: spike-run tool absent; one cold inline run, 12.9 s, three `in sys.modules` checks False |
| A10 | cisternal runs sync tool bodies on the event loop, so FastMCP worker threads cannot be relied on (replaces the earlier wrong thread assumption) | async twins unnecessary | VERIFIED | read: cisternal src/cisternal/registration/compose.py:130-133; shim.py:62-64 |
| A11 | `prolix.__version__` is "0.1.0" while pyproject says 1.0.0, so the record reads `importlib.metadata` | record shows a wrong version | VERIFIED | read: src/prolix/__init__.py:3; pyproject.toml:7 |
| A12 | `myxcel submit REMOTE PROJECT SCRIPT_PATH` accepts `--dry-run`/`--respect-directives`, requires a local script, and maps in-tree paths to the remote | contract changes | VERIFIED | read: myxcel src/myxcel/cli.py:1689,1792-1830 |
| A13 | the project name `prolix` and remote `engaging` exist in the owner's myxcel config | `submit_argvs` defaults change | UNVERIFIED | deferred: remote profiles live in the owner's `~/.config/myxcel`, not read on purpose; both are parameters |
| A14 | `bonded_energy_fn_from_bundle` is the bonded-only energy | breakdown drops to total only | VERIFIED | read: src/prolix/api/bundle_md.py:223,579-604 |
| A16 | cisternal's CLI path calls tool bodies synchronously, so async tools need a separate MCP registry | one async tool would return an unawaited coroutine on the CLI | VERIFIED | read: cisternal src/cisternal/registration/wired.py:103 (`apply_recovery_sync`) |
| A17 | dt units are AKMA (1 unit = 48.888 fs) in `run_nve`/`run_nvt_nose_hoover` | dt conversion changes | VERIFIED | read: src/prolix/simulate.py:56; src/prolix/physics/simulate.py:283,310 |
| A18 | CI runs `uv sync --extra dev` and `uv run ty check` | CI changes unneeded | VERIFIED | read: .github/workflows/ci.yml:36,59 |
| A19 | `format_success` is required for `--json` output; `json_option` only adds a flag | `--json` would be inert | VERIFIED | read: cisternal src/cisternal/registration/cli_contract.py:118-133 |
| A15 | cisternal: `@tool(registry=...)`, `wire(server, app, registry, expected, cli_contract)`, no middleware from `wire`, fork unsupported; `src/prolix` uses no `multiprocessing` | wiring or spawn policy changes | VERIFIED | read: cisternal src/cisternal/registration/decorator.py:54-65; wired.py:408-420; src/prolix/tiling/planner.py:9 is the only `fork` hit (a comment) |
| A20 | a package `src/prolix_agent_entry/` with `__init__.py` is shipped by the existing `packages.find where=["src"]`, and `[tool.setuptools.package-data]` ships `job_scripts.txt` | scripts would not resolve after install | UNVERIFIED | deferred: no build run (never run a build here); read: pyproject.toml:50-51; Task 1 gate inspects the built wheel |
| A21 | `myxcel push` ships only files `git ls-files --cached --others --exclude-standard` lists (minus excludes), and `myxcel submit` uploads nothing | artifact location is free | VERIFIED | read: myxcel src/myxcel/rsync.py:173-182,69-80; myxcel src/myxcel/cli.py:1792-1830 |
| A22 | `scripts/slurm/agent_jobs/x/job.slurm` is not git-ignored in prolix; `outputs/` is | artifact path is dropped by push | VERIFIED | read: .gitignore:27; inline `git check-ignore -v scripts/slurm/agent_jobs/x/job.slurm` exited 1 (no match) |
| A23 | `scripts/benchmarks/prolix_vs_openmm_speed.py` exists with argparse flags `--sweep`, `--warmup` (no `--n`) | golden args and allowlist content change | VERIFIED | read: scripts/benchmarks/prolix_vs_openmm_speed.py:228-231 |
| A24 | `mit_normal_gpu` with `gpu:l40s:1`, 16 cpus, 64G is a header prolix scripts use; its walltime limit is 2 h or more | `gpu_l40s` max time changes | VERIFIED | read: scripts/slurm/prof_ensemble_apply_probe.slurm:2-7; read: /home/marielle/projects/myxcel/.praxia/docs/reference/260714_lab-demo-script.md:63-73 (mit_quicktest 14 min cap, `gpu` preset mit_normal_gpu 05:45:00 with l40s:1, 16 cpus, 64G) |
| A25 | `myxcel submit --auto-requeue` is rejected together with `--respect-directives`; preemptable jobs without a resumable runner restart from scratch | a requeue flag could rescue a preemptable preset | VERIFIED | read: myxcel src/myxcel/cli.py:1699,1761-1762,2087-2089; prolix src/prolix/physics/simulate.py:141-197,259-328 |
| A26 | `myxcel submit` prints a `slurm_id: <n>` line on success, and `job_status(remote, slurm_id)` exists as a myxcel MCP tool | id parse or follow-up pointer changes | VERIFIED | read: myxcel src/myxcel/cli.py:2175; myxcel src/myxcel/tools.py:116-122 (output format may change; parse falls back to `null`) |
| A27 | with `JAX_PLATFORMS=cpu` set before `import prolix`, only the cpu backend is initialised | CPU policy fails | VERIFIED | deferred: probe, not spike-run (spike tool absent): printed `cpu [CpuDevice(id=0)]`, `True ['cpu']` on the worktree src with the main checkout venv |
| A28 | cyclopts' own usage-error exit is version-dependent (orchestrator probe 261123: 4.21.1 and 4.25.3 exit 1, 5.5.2 exits 2), so the entry maps `CycloptsError` -> 2 itself; a control test with the wrapper removed must not pin 2 on 4.x; xtrax.agent exits 2 (cache) and 1 (cisternal missing) | exit table changes | UNVERIFIED | deferred: probe, not spike-run, on cyclopts 5.4.2 in a scratch venv (all three exit 2), not prolix's locked version; read: xtrax src/xtrax/agent.py:93,98 (PR #284 worktree); the §11 exit-code test guards the locked version |
| A29 | `run_nve` ignores any temperature argument (300 K, key 0) and takes `mass=`; `run_nvt_nose_hoover` honours `temperature` and `mass` | md_smoke arguments change | VERIFIED | read: src/prolix/physics/simulate.py:259-328 (284-286 for the hard-coded 300 K) |
| A30 | a mass array of shape `(n_real,)` is accepted by `run_nve`/`run_nvt_nose_hoover` | md_smoke mass handling changes | UNVERIFIED | deferred: no run here; read: src/prolix/api/bundle_md.py:109-117 (`masses_with_prefix` returns a `(prefix,)` mass array); covered by the 3-atom md_smoke CI test |
| A31 | `myxcel submit` runs `mkdir -p` only on its own `remote_log_dir` (default `outputs/logs/slurm`), and the pushed script's directory exists remotely so a log path beside it opens | log path under a missing dir: job fails with no log | UNVERIFIED | read: /home/marielle/projects/myxcel/src/myxcel/cli.py:2012-2014 (mkdir of `remote_log_dir` only); deferred: that push creates `scripts/slurm/agent_jobs/<name>/` remotely follows from A21 but is not run here; Task gate adds a rendered-script assertion that `--output`/`--error` sit under the artifact's directory |

## 13. Risks

| Risk | Mitigation |
|---|---|
| Debt #3586 (second NameError, NL path) | tools stay on the dense path; test asserts no `neighbor` argument |
| Caps wrong on other machines | lower-only constants, calibration table in the PR, slow wall-clock test |
| `submit_job` reachable by an agent | dry-run default, `confirm`, only rendered scripts with matching sha256, argv list without a shell |
| Rendered script drifts from `scripts/slurm` conventions | golden file reviewed against `b1_init_exec.slurm`; presets in one table |
| Loader copy drifts from the script | parity test |
| `prolix --help` pays the 13 s import | documented; lazy `__init__` is the follow-up |
| Entry package name collides or is dropped from the wheel | Task 1 wheel gate; the subprocess test (§11) fails if `prolix` runs without the entry |
| `agent_jobs/` clutter, `myxcel push` ships the whole tree | documented in `notes`; dirs are untracked and human-deleted |
| Preempted or timed-out job restarts from step 0 | no preemptable preset; 2 h cap; resumable runner is the follow-up |
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
- C8 accepted: template sources `_uv_env.sh` like `l3_smoke.slurm`, header provenance corrected, `rc` capture, stub-`uv` failure test. C9 accepted: conftest CPU guard (revised in round 2: F1).
- C10 accepted: `periodic` dropped. C11 modified: H-free PDBs warn and record `n_hydrogen` rather than refuse, because the probes show the loader adds no hydrogens and the fixtures are H-free. C12 recorded in A6/A8/A9 (old venv, jax 0.10.2). C13 accepted: out_dir root, `.tmp.npz`.
- C14 accepted: minimize assertion, `bonded` defined (exact term list confirmed from a read in Task 2). C15 accepted: exit-code table. C16 accepted: base floor unchanged, regex handles extras.
- C17 accepted: Task 0 dropped, depends on #46; debt #3586 scoped to the NL path, tools never pass `neighbor`.

Round 2 (challenger F1-F2, M1-M4, m1-m3), verified at source.
- F1 accepted: `import prolix` initialises JAX (`cmap.py:39`), so `ensure_cpu()` in `main()` was too late. Console scripts now point at `prolix_agent_entry` (top-level package, no prolix import) which sets `JAX_PLATFORMS=cpu` first (D10, §3, §7); `assert_cpu()` fails closed for other callers. The "`import prolix` leaves backends uninitialised" test is replaced by a subprocess test of the real entry with a poisoned `JAX_PLATFORMS=cuda` plus a bypass control (§11). Lazy `prolix/__init__`/`cmap` stays a follow-up.
- F2 accepted: `outputs/` is ignored and `myxcel push` ships only non-ignored files while `submit` uploads nothing (A21). Artifact moves to `scripts/slurm/agent_jobs/<name>/`; `submit_job` runs `myxcel push` then `myxcel submit`; the pushed-set test uses myxcel's own `git ls-files` command with a failing control for `outputs/` (§6, §11).
- M1 accepted: `gpu_h200_preempt` removed; presets are `cpu_quick` and `gpu_l40s` (non-preemptible). Preemptable preset needs a resumable runner (follow-up; `--auto-requeue` is rejected with `--respect-directives`, A25).
- M2 accepted: exit table rewritten with one test per row; code 1 and 2 overlaps documented with the stderr prefix that separates them; cyclopts exit 2 probed (A28); `AgentExtraMissing` exits 1 through `prolix_agent_entry`.
- M3 accepted: `job_scripts.txt` format, initial content (`prolix_vs_openmm_speed.py`), loader and tests specified; golden args corrected to the script's `--sweep` flag (A23).
- M4 accepted: `temperature_k` is NVT-only (NVE with it is `InputError`); both integrators receive `mass=bundle.masses[:n_real]` (A29, A30).
- m1 accepted: D7/§4 cite A10 (event loop) and A16 (sync CLI). m2 accepted: `slurm_id` parsed from submit stdout, `follow` points to myxcel `job_status` (A26). m3 accepted: what `--dry-run` proves and does not prove is stated in §6.
- Orchestrator note (post round 2): usage-error exit 2 now comes from a `CycloptsError` wrapper, not cyclopts' default, which is 1 on 4.x and 2 on 5.x (A28).

Round 3 (converged; no FATAL/MAJOR). Minors applied:
- R3-m1 accepted: §11 golden renamed `job_gpu_h200.slurm` -> `job_gpu_l40s.slurm` (h200 preset removed in round 2).
- R3-m2 accepted: `--output/--error` now point beside the script at `scripts/slurm/agent_jobs/<name>/%j.<name>.{out,err}` (a pushed, non-ignored path), not gitignored `outputs/`; myxcel's `mkdir -p` covers only its own `remote_log_dir` (`myxcel cli.py:2012-2014`), so the template does not rely on it; claim tagged A31 (UNVERIFIED, deferred) with a golden assertion on the log paths.
- R3-m3 accepted: the agent-extra check moved into `prolix_agent_entry._require_agent_extra()` (`find_spec`, before any prolix import); exit 1 unchanged (§3).
- R3-m4 accepted: A24 VERIFIED from myxcel `260714_lab-demo-script.md:63-73`; the 2 h cap sits under the 5:45 partition limit.
