"""XTRAX_GIT_SHA / .git_sha file beat a missing .git on cluster scratch.

The env var is XTRAX_GIT_SHA, not PROLIX_GIT_SHA. It was renamed when
scripts/profiling/ moved upstream into xtrax.profiling, and this module is
exercising xtrax's implementation -- see xtrax/profiling/record.py, which reads
XTRAX_GIT_SHA and documents the rename in its own module docstring.

Getting this wrong does not raise: _capture_git_sha() falls through to the
.git_sha file and then to a `git rev-parse` that finds nothing on a cluster
scratch dir with no .git, so records silently carry a blank sha. That is exactly
what these two tests exist to catch, and what they caught on job 21789004.

The SLURM wrappers export BOTH names, because scripts/experiments/
prof_ensemble_step.py and scripts/explore/prof_flash_tile_checkpoint.py each
carry their own _git_sha() that still reads PROLIX_GIT_SHA.
"""

from __future__ import annotations

from xtrax.profiling import record as rec


def test_capture_git_sha_from_env(monkeypatch):
    monkeypatch.setenv("XTRAX_GIT_SHA", "abc123deadbeef")
    assert rec._capture_git_sha() == "abc123deadbeef"


def test_capture_git_sha_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("XTRAX_GIT_SHA", raising=False)
    sha_file = tmp_path / ".git_sha"
    sha_file.write_text("cafebabeface\n")
    monkeypatch.setattr(rec, "_REPO_ROOT", tmp_path)
    assert rec._capture_git_sha() == "cafebabeface"
