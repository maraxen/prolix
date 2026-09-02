"""PROLIX_GIT_SHA / .git_sha file beat a missing .git on cluster scratch."""

from __future__ import annotations

from xtrax.profiling import record as rec


def test_capture_git_sha_from_env(monkeypatch):
    monkeypatch.setenv("PROLIX_GIT_SHA", "abc123deadbeef")
    assert rec._capture_git_sha() == "abc123deadbeef"


def test_capture_git_sha_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("PROLIX_GIT_SHA", raising=False)
    sha_file = tmp_path / ".git_sha"
    sha_file.write_text("cafebabeface\n")
    monkeypatch.setattr(rec, "_REPO_ROOT", tmp_path)
    assert rec._capture_git_sha() == "cafebabeface"
