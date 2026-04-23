# UV workspace root (sibling **prolix** only)

These files are copied to the **parent** of the prolix checkout — for example `~/projects/pyproject.toml` and `~/projects/uv.lock` when the repo lives at `~/projects/prolix`.

**proxide** is installed from **PyPI** (manylinux wheels for x86_64 and aarch64). A local `../proxide` tree is optional and only needed if you are hacking proxide itself; `just push-engaging` may still rsync it for convenience.

Install on the cluster (or locally):

```bash
cd ~/projects
uv sync --extra cuda --extra dev --package prolix
```

OpenMM (optional extra) is included in the Engaging SLURM scripts as `--extra openmm`.

To refresh `uv.lock` after dependency changes, from the prolix repo run `bash scripts/sync_workspace_lock.sh`, then commit the updated `workspace/uv.lock`.

The authoritative lock for this layout is **`workspace/uv.lock`** (deployed to the parent directory). A `uv.lock` at the prolix repo root is not used for sibling workspace installs and may be stale.

### Co-developing proxide (editable local checkout)

After a normal `uv sync`, replace the PyPI install with your tree, e.g. `uv pip install -e ../proxide` from `~/projects`, or use a temporary `[tool.uv.sources]` override in a **local-only** branch (not committed) so `uv lock` still matches CI.
