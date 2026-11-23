---
title: 'prolix CLI and MCP server: quick in-process operations, long runs rendered as jobs'
description: 'Implementation spec for debt #3555: a cisternal-wired prolix CLI and MCP server (src/prolix/agent) exposing seconds-scale operations in-process on CPU and rendering or submitting long runs through myxcel; built on xtrax.agent.'
status: draft
task_id: 261123_xtrax-entrypoint-cli-mcp
date: '261123'
---
# prolix CLI and MCP server (debt #3555)

Base: prolix `origin/main` 5ce098d. Builds on the converged `261123_xtrax-agent-entry.md` (xtrax PR #284: `xtrax.agent.run_cli`
/ `serve`; its §7 lists what this spec owns). Owner decision (fixed): only seconds-scale operations run in-process; long
runs are rendered as a job script or submitted via myxcel, never run inside a call.

## 1. Problem, and what recon got wrong

prolix has no CLI, `__main__` or `[project.scripts]`; `src/prolix/viewer_server.py:268` is its only argparse. Reading code
changes the design in four ways:

1. **`MolecularBundle.from_pdb` cannot be the loader.** It validates `forcefield` and then ignores it; the parmed path yields
   empty parameters for a PDB, as its own comment says (`types/bundles.py:277-297,336-345`). Its docstring ("AMBER14 parameters") is false.
2. **The working loader is unpackaged**: `scripts/benchmarks/_b1_paramize.py:41` (proxide `parse_structure(parameterize_md=True)`,
   ff19SB, `ExclusionSpec.from_protein`); `packages.find where=["src"]` (pyproject.toml:50-51).
3. **`make_bundle_from_system` raises `NameError` on origin/main**: `physics/system.py:874-876` read `use_size_buckets`, which is not
   a parameter (`_next_bucket` at `:390` and `tests/physics/test_bundle_factory.py:94` expect it). Every bundle build fails, including
   `tests/api/test_hp2_energy_fn_from_bundle.py`. A library bug independent of this spec; prerequisite Task 0.
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
| D6 | One xtrax floor in `[project.dependencies]` and the `agent` extra; `check_xtrax_pin.py` reads pyproject | two floors (stale-floor defect recurs) |

## 3. Layout and entry points

```
src/prolix/agent/__init__.py   require_agent_extra() only (aminx/agent/__init__.py pattern); imports nothing heavy
src/prolix/agent/errors.py     AgentError > InputError, RefusalError
src/prolix/agent/_runtime.py   ensure_cpu(), cap constants, run lock, versions(), sha256(), atomic_write(), write_record()
src/prolix/agent/system.py     load_system_bundle(path, ff_name, periodic): copy of _b1_paramize.paramize_pdb_to_bundle
src/prolix/agent/ops.py        pure functions, one per operation (no cisternal import); golden tests call these
src/prolix/agent/render.py     render_job_script(), submit_argv() (no cisternal import)
src/prolix/agent/tools.py      REGISTRY="prolix", TOOL_NAMES, @cisternal.tool wrappers over ops/render
src/prolix/agent/cli.py        require_agent_extra() on its FIRST line; cyclopts app; wire(None, app, ...); main()
src/prolix/agent/mcp.py        require_agent_extra() on its first line; build_server(); main()
```

`pyproject.toml`: `[project.scripts] prolix = "prolix.agent.cli:main"`, `prolix-mcp = "prolix.agent.mcp:main"`;
`agent = ["xtrax[plugin]>=X,<0.5"]` (cisternal, cyclopts, fastmcp come transitively); `xtrax>=X,<0.5` in dependencies. `X` = first
xtrax release containing `xtrax.agent` (after 0.4.1b9; A1). `cli.py` and `mcp.py` call `require_agent_extra()` before any cisternal,
cyclopts or fastmcp import (they wire at import time), so a missing extra prints the `prolix[agent]` hint instead of a
`ModuleNotFoundError`. `prolix/__init__.py` does not import `prolix.agent`; `ops/render/_runtime` import none of the three. Heavy imports
(`jax`, `proxide`, `prolix.api`) sit inside operation bodies. `import prolix` costs ~13 s cold (A9), so `prolix --help` pays it;
a lazy `prolix/__init__` is a follow-up.

```python
def main() -> None:                 # cli.py, after the require/import lines
    _runtime.ensure_cpu()           # FIRST: run_cli starts and dispatches in one call, so nothing can run between
    agent.run_cli(app)              # exit 2 bad cache config, exit 1 missing xtrax[plugin]
def main() -> None:                 # mcp.py
    _runtime.ensure_cpu(); agent.serve(build_server())
```

`build_server()` mirrors `aminx.agent.mcp.build_server`: `FastMCP("prolix", instructions=...)`, `cisternal.wire(server, registry="prolix",
expected=list(TOOL_NAMES))`, `add_middleware(CisternalMiddleware(adapter=PassthroughAdapter(), reraise=True))`. `cli.py`:
`wire(None, app, registry="prolix", expected=TOOL_NAMES, cli_contract=CONTRACT)` at import, so a renamed or dropped tool fails at
import (demistify `cli.py:1013`). `CONTRACT`: `exit_codes={InputError: 4, RefusalError: 3}`, `json_option()`. Tool bodies return dicts and
raise `AgentError`; they never print (cisternal rule 2).

## 4. Operations

In-process tools take `out_dir` (required) and write `<tool>.json` (run record, §5) there, plus named side files. Arrays never go
inline: they are written to `.npz` and the result carries path and sha256. Shared arguments: `pdb`, `ff` (default
`protein.ff19SB.xml`, a bundled proxide name or an absolute path, as `_resolve_ff_path` does), `overwrite=false`.

| Tool | Does | Library call | Side files | Result keys |
|---|---|---|---|---|
| `load_system` | parameterise, validate, summarise | `system.load_system_bundle` | none | real/padded atoms, bucket idx, term counts, charge sum, `has_pbc`, ff path |
| `inspect_params` | parameter rows for one term (`bond,angle,dihedral,improper,charge,lj`), `limit<=200` | same bundle | none | rows, `truncated` |
| `energy_breakdown` | total, bonded, nonbonded, force stats | `energy_fn_from_bundle`, `bonded_energy_fn_from_bundle`; nonbonded = total - bonded | none | `total`, `bonded`, `nonbonded_by_difference`, median/max force (kcal/mol/Å) |
| `minimize` | bounded steepest descent | `run_minimization(..., sd_steps=n)` | `minimized.npz` | energy before/after, `sd_steps` |
| `md_smoke` | short NVE or NVT Nose-Hoover | `run_nve` / `run_nvt_nose_hoover` | `final.npz` | `finite`, start/end energy, steps, dt, ensemble, `seed:"fixed 0"`; no conservation claim |
| `render_job` | write an sbatch script (§6) | `render.render_job_script` | `job.slurm` | path, sha256, resources, `submit_argv` |
| `submit_job` | `myxcel submit` (§6) | `render.submit_argv` + `subprocess.run` | none | `dry_run`, argv, exit code, stdout, stderr |

Positions: `api.bundle_md.active_positions(bundle)` (real atoms), as `_b1_paramize.bundle_force_stats` does; masses `bundle.masses[:n_real]`.

**Hard caps** (constants in `_runtime.py`; a call may lower, never raise; no CLI override). Placeholders, which Task 2 must lower until
the largest allowed combination runs in about 60 s on 8 CPU threads (dense energy scales ~N², so measure at the cap, not at 327 atoms):
`MAX_ATOMS=1000`, `MAX_MIN_STEPS=200`, `MAX_MD_STEPS=500`, `MAX_DT_PS=2e-3`. The atom check counts `ATOM`/`HETATM` lines before
parameterising and checks `n_real` again after. A violation raises `RefusalError` naming the limit, the request, and `render_job`.
`InputError` for: missing file, non-`.pdb`, unknown ff name, existing record without `overwrite`, periodic system without a box,
`dt` above cap, non-finite input energy, and a loaded bundle whose `bond_params` are all zero (the `from_pdb` failure mode).
Refusals precede any write. One process-wide `threading.Lock` serialises in-process operations (A10); a wait over 30 s raises
`RefusalError("busy")`.

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
- `script`: repo-relative path under `scripts/`, ending `.py`, existing and resolving inside `repo_root` (`..`, absolute and symlink escapes
  refused). `repo_root` defaults to the nearest ancestor of cwd with `pyproject.toml` `name = "prolix"` and `scripts/slurm/_common_env.sh`,
  else `InputError` (an installed wheel has no `scripts/`).
- Output: `<repo_root>/outputs/agent_jobs/<name>/job.slurm` plus `render_job.json` (`outputs/` is gitignored, `.gitignore:27`). Writing
  inside the checkout is what lets `myxcel submit` map the path; this is the one exception to D4.
- `args`: strings, each `shlex.quote`d. `preset` in a closed table in `render.py`: `cpu_quick` (`mit_quicktest`, 2 cpus, 4G, 10 min) or
  `gpu_h200_preempt` (`mit_preemptable`, `--gres=gpu:h200:1`, 8 cpus, 256G; from `b1_init_exec.slurm`). `time` `HH:MM:SS`, default and
  maximum `12:00:00`. `name` `[A-Za-z0-9_-]{1,40}`.
- Script body: `#!/bin/bash`, preset `#SBATCH` lines with `--job-name=prolix_<name>` and `--output/--error=outputs/logs/slurm/%j.<name>.{out,err}`,
  `set -euo pipefail`, `source scripts/slurm/_common_env.sh`, `uv run python <script> <quoted args>`, `echo "=== <name> exit=$? ==="`. No `bth run`
  (a run that yields a finding needs its own tracked script and sidecar).
- Result: `submit_argv = ["myxcel","submit",<remote>,"prolix",<repo-relative script path>,"--respect-directives"]` (remote default `engaging`) and
  `notes`: sync the checkout to the remote first (the script and `scripts/` must exist there), then submit.

`submit_job(name, remote="engaging", confirm=false, repo_root=None)`: resolves `<repo_root>/outputs/agent_jobs/<name>/job.slurm` and refuses unless its
`render_job.json` exists with a matching script sha256 (only rendered scripts submit); appends `--dry-run` unless `confirm`; runs
`subprocess.run(argv, shell=False, timeout=120, capture_output=True)`; returns argv, exit code, stdout, stderr; parses no job id. Missing
`myxcel` raises `InputError` with the argv to run by hand. `subprocess` is spawn-safe; cisternal forbids only `fork` multiprocessing.

## 7. Process, device and multiprocessing policy

- Startup is `xtrax.agent` only; `main()` never calls `jax_cache.configure()` or `cisternal.init()` itself.
- `ensure_cpu()`: `jax.config.update("jax_platforms","cpu")`, first statement of both `main()`s (A6), never edits `os.environ`. Each in-process
  op re-asserts `jax.default_backend() == "cpu"` and raises `RefusalError` if a backend was already initialised elsewhere (fails closed; a
  GPU box cannot silently run an in-process tool on GPU). `xtrax/jax_cache.py` has no `devices(`/`default_backend`/`xla_bridge` call (A6);
  a subprocess test checks that `import prolix` leaves `jax.lib.xla_bridge.backends_are_initialized()` false. Pallas stays default-off
  (`pallas_kernels.py:81-92`, enabled only by `PROLIX_USE_PALLAS`); the record stores the flag. x64 is untouched (tests flip it per fixture,
  `tests/physics/conftest.py:66-80`). No `device=` argument is offered.
- No `multiprocessing`: `src/` has none (A15); a test greps `src/prolix/agent` for `multiprocessing`/`os.fork`.
- `MAX_ATOMS` bounds the dense `N x N` exclusion matrices (`bundle_md.py` docstring: ~4.4 GB at N=23,558).

## 8. Packaging

`scripts/ci/check_xtrax_pin.py`: replace the hard-coded `FLOOR = "0.4.0a6"` (`:14`) with the floor parsed from pyproject (`xtrax>=X,<U`); CI runs
it (`ci.yml:61-63`). `tests/scripts/test_check_xtrax_pin.py` asserts parsed floor == the `agent` extra's floor. `uv.lock` regenerates in the same PR.
`CHANGELOG.md` entry; README "Agent CLI/MCP" section (~10 lines). Not added: `agent_plugin.json`, skills, `.mcp.json`, PDB output (`.npz` only),
GPU in-process, a lazy `__init__`, long-run executors or checkpointing, migrating `_b1_paramize`.

## 9. Old vs new

| Concern | Today | After |
|---|---|---|
| Entry points | none | `prolix`, `prolix-mcp` |
| PDB loading | `from_pdb` (no parameters, silent) or an unpackaged script | `load_system_bundle`, fails closed |
| Bundle factory | `NameError` on every call | fixed (Task 0) |
| Long runs | hand-edited `.slurm`, manual submit | `render_job` + `submit_job` (dry-run default) |
| Outputs | `outputs/` by convention | `out_dir` + JSON record (sha256s, versions, cache setting) |
| xtrax floor | pyproject `0.4.1b2`, checker `0.4.0a6` | one floor, checker reads pyproject |

## 10. Tasks (one PR; commits in order)

- **Task 0, fix `make_bundle_from_system`** (`physics/system.py`): add `use_size_buckets: bool = True`; pass it to the atom and water
  `_next_bucket` calls, with `exact_n_*` as `:875-876` assume. Gate: `pytest tests/api/test_hp2_energy_fn_from_bundle.py
  tests/physics/test_bundle_factory.py -q`. ~10 LOC. May be split into its own PR; Tasks 1-5 rebase on it.
- **Task 1, scaffold**: `agent/{__init__,errors,_runtime}.py`, pyproject scripts/extra/floor, `check_xtrax_pin.py`, lock. Gate: `import prolix.agent`
  loads none of cisternal/fastmcp/cyclopts; pin test green. ~150 LOC.
- **Task 2, operations and loader**: `system.py`, `ops.py`, caps calibration (table goes in the PR). Gate: round-trip tests. ~400 LOC.
- **Task 3, render/submit**: `render.py`. Gate: golden test. ~200 LOC.
- **Task 4, wiring**: `tools.py`, `cli.py`, `mcp.py`, README, changelog. Gate: parity tests. ~250 LOC.
- **Task 5, tests** (§11), fixture `tests/agent/data/mini.pdb` = residues 1-3 of `data/pdb/1CRN.pdb` (`awk '$1=="ATOM" && $6<=3'` plus `END`;
  20 atoms; parameterises in the probe, A8). ~500 LOC.

## 11. Tests (`tests/agent/`; narrow local runs, CPU)

Tests needing the extra use `pytest.importorskip("cisternal")`.
- **Registry parity:** `EXPECTED_TOOLS` (the seven names) == `set(TOOL_NAMES)` == the cisternal registry snapshot for `prolix` == `build_server()` tool names
  == cyclopts command names. Control: a registry missing one name makes `wire(expected=)` raise.
- **Missing extra:** with `cisternal` import blocked (`builtins.__import__` raising `ModuleNotFoundError(name="cisternal")`), importing `prolix.agent.cli`
  raises the `prolix[agent]` `ImportError` hint, not the raw error. Same for `mcp`.
- **Round trip, all tools, one test sharing one loaded bundle** (compile cost is per energy function; no persistent cache in tests): op call and CLI call
  (`cli.app([...,"--json"], result_action="return_value")`) on `mini.pdb` with a tmp `out_dir`. Assert record exists, schema, PDB sha256 equals `hashlib`,
  `jax_cache` key present, output hashes match files, no `*.tmp` left.
- **Golden vs library** (same process, same flags): `energy_breakdown.total` and `.bonded` equal the direct `energy_fn_from_bundle` /
  `bonded_energy_fn_from_bundle` values (`rtol=1e-6`); `minimize` positions equal `run_minimization(ef, x, sd_steps=n)`; `md_smoke` final positions
  `allclose` `run_nve`; `load_system_bundle` energy equals `_b1_paramize.paramize_pdb_to_bundle` (importable via `pythonpath=["."]`, pyproject). Controls:
  moving one atom 0.1 Å changes the total beyond tolerance; a `from_pdb` bundle (after Task 0) is refused by the all-zero-parameter check.
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
- **Budget:** one compile-bearing round-trip test plus fast refusal/render tests in CI; Task 5 records the measured wall time (A8: ~13 s load and ~13 s
  first energy compile per probe) and marks any compile-bearing test beyond a 90 s total `slow`. A `slow` test runs the largest allowed combination under 120 s.

## 12. Assumption ledger

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | the next xtrax release after 0.4.1b9 ships `xtrax.agent` and `xtrax[plugin]` | pin and lock cannot be written | UNVERIFIED | deferred: owner's release process; xtrax PR #284 is unreleased |
| A2 | `from_pdb` applies no force-field parameters | loader choice changes | VERIFIED | read: src/prolix/types/bundles.py:277-297,336-345 |
| A3 | `make_bundle_from_system` raises `NameError` at the `use_size_buckets` reference | Task 0 changes | VERIFIED | read: src/prolix/physics/system.py:672-676,874-876 |
| A4 | `run_minimization` uses only `sd_steps`; `run_nve`/`run_nvt` use key 0 and return positions only | caps and record fields change | VERIFIED | read: src/prolix/physics/simulate.py:141-197,259-328 |
| A5 | `scripts/` is not packaged | the loader could be imported from scripts | VERIFIED | read: pyproject.toml:50-51 |
| A6 | `jax.config.update("jax_platforms","cpu")` after `import jax` and before device use selects CPU; xtrax `jax_cache` initialises no backend | CPU policy needs an env var set before import | UNVERIFIED | deferred: spike-run tool absent here (scripts/loop/adversarial_metrics.py not in this workspace); an inline probe printed `[CpuDevice(id=0)]` on jax 0.10.2 from the main checkout's older venv, not the worktree lock; read: xtrax src/xtrax/jax_cache.py (no `devices(`/`default_backend` call) |
| A7 | `jax_cache.current()` returns the applied setting; `as_record()` is JSON-ready | record key changes | VERIFIED | read: xtrax src/xtrax/jax_cache.py:223-240,827-836 |
| A8 | `mini.pdb` (20 atoms) and 1CRN (327 atoms) parameterise through `_b1_paramize` once Task 0 is applied, with finite energies and the timings used for calibration | fixture and caps change | UNVERIFIED | deferred: spike-run tool absent; inline probe on a patched temp copy (CPU, 4 threads, older venv): 20 atoms load 13.5 s, grad 13.2 s, 50 sd 2.4 s, 200 NVE 2.7 s; 327 atoms 9.3 s, 14.7 s, 3.1 s, 3.7 s |
| A9 | `import prolix` takes ~13 s and imports none of cisternal, fastmcp, cyclopts | lightness test and `--help` note change | UNVERIFIED | deferred: spike-run tool absent; one cold inline run, 12.9 s, three `in sys.modules` checks False |
| A10 | FastMCP runs sync tools on worker threads | the lock is harmless overhead | UNVERIFIED | deferred: depends on the fastmcp version cisternal resolves; the lock stays either way |
| A11 | `prolix.__version__` is "0.1.0" while pyproject says 1.0.0, so the record reads `importlib.metadata` | record shows a wrong version | VERIFIED | read: src/prolix/__init__.py:3; pyproject.toml:7 |
| A12 | `myxcel submit REMOTE PROJECT SCRIPT_PATH` accepts `--dry-run`/`--respect-directives`, requires a local script, and maps in-tree paths to the remote | contract changes | VERIFIED | read: myxcel src/myxcel/cli.py:1689,1792-1830 |
| A13 | the project name `prolix` and remote `engaging` exist in the owner's myxcel config | `submit_argv` defaults change | UNVERIFIED | deferred: remote profiles live in the owner's `~/.config/myxcel`, not read on purpose; both are parameters |
| A14 | `bonded_energy_fn_from_bundle` is the bonded-only energy | breakdown drops to total only | VERIFIED | read: src/prolix/api/bundle_md.py:223,579-604 |
| A15 | cisternal: `@tool(registry=...)`, `wire(server, app, registry, expected, cli_contract)`, no middleware from `wire`, fork unsupported; `src/prolix` uses no `multiprocessing` | wiring or spawn policy changes | VERIFIED | read: cisternal src/cisternal/registration/decorator.py:54-65; wired.py:408-420; src/prolix/tiling/planner.py:9 is the only `fork` hit (a comment) |

## 13. Risks

| Risk | Mitigation |
|---|---|
| Task 0 changes library behaviour | ~10-line diff, existing tests cover it, separable PR |
| Caps wrong on other machines | lower-only constants, calibration table in the PR, slow wall-clock test |
| `submit_job` reachable by an agent | dry-run default, `confirm`, only rendered scripts with matching sha256, argv list without a shell |
| Rendered script drifts from `scripts/slurm` conventions | golden file reviewed against `b1_init_exec.slurm`; presets in one table |
| Loader copy drifts from the script | parity test |
| `prolix --help` pays the 13 s import | documented; lazy `__init__` is the follow-up |
| xtrax release slips | PR stays draft; rollback is reverting it (only Task 0 touches library code) |

## 14. Delivery

One PR to prolix `main` off 5ce098d, opened after the xtrax release (A1) and depending on nothing else. Commit order: Tasks 0-5. Gates are narrow
local runs (`pytest tests/agent -q`, `tests/physics/test_bundle_factory.py`); the full suite runs in CI only. The PR body lists the calibration table
and the pin value. Reviewer reads first: `ops.py` caps, `render.py` quoting and presets, `submit_job` guards. Follow-up debts to file: lazy
`prolix.__init__`; stale `scripts/slurm/install_xtrax_wheel.slurm:99-107` assertion (`0.4.0a7`); `from_pdb` docstring versus behaviour.
