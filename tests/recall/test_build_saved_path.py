"""`synapt recall build` must print the store it actually wrote.

The build writes with an explicit project directory (derived from the working directory), and an explicit project
directory suppresses the environment's store roots. Two of its printed lines asked the zero-argument resolver instead,
which reads SYNAPT_RECALL_ROOT / GRIPSPACE_ROOT: with the environment naming a different store, "Saved to:" named a
store the build never touched while the files changed in another.
"""

import argparse
from pathlib import Path
from unittest import mock

import pytest

from conftest import assistant_entry, user_text_entry, write_jsonl
from synapt.recall import cli
from synapt.recall.core import project_index_dir


def _tree(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else None


@pytest.fixture
def stores(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    envroot = tmp_path / "envroot"
    envroot.mkdir()
    source = tmp_path / "src"
    source.mkdir()
    write_jsonl(source / "s1.jsonl", [
        user_text_entry("deploy the kubernetes rollout canary staging", uuid="u1", ts="2026-03-01T10:00:00Z"),
        assistant_entry(text="rollout canary staging manifest helm chart release", uuid="a1", ts="2026-03-01T10:00:30Z"),
    ])
    written = project_index_dir(project.resolve())
    named_by_env = envroot.resolve() / ".synapt" / "recall" / "index"
    # the control that makes every row below mean something: two different stores, both ours
    assert written != named_by_env and str(written).startswith(str(tmp_path.resolve())), (written, named_by_env)
    return dict(project=project, envroot=envroot, source=source, written=written, named_by_env=named_by_env)


def _build(stores, capsys, env_var, legacy=None):
    import os
    capsys.readouterr()
    args = argparse.Namespace(source=[str(stores["source"])], hf=None, chatgpt_archive=None,
                              no_embeddings=True, incremental=False)
    patches = [mock.patch("synapt.recall.cli.Path.cwd", return_value=stores["project"]),
               mock.patch.dict(os.environ, {env_var: str(stores["envroot"])} if env_var else {}, clear=False)]
    if legacy is not None:
        patches.append(mock.patch.object(cli, "_check_legacy_index", return_value=legacy))
    for p in patches:
        p.start()
    try:
        cli.cmd_build(args)
    finally:
        for p in reversed(patches):
            p.stop()
    return capsys.readouterr().out


@pytest.mark.parametrize("env_var", ["GRIPSPACE_ROOT", "SYNAPT_RECALL_ROOT"])
def test_saved_to_names_the_store_whose_files_changed_not_the_one_the_environment_names(stores, capsys, env_var):
    env_before = _tree(stores["envroot"])
    out = _build(stores, capsys, env_var)

    written, named_by_env = stores["written"], stores["named_by_env"]
    assert (written / "recall.db").exists(), "the build did not write where the test expects"
    assert _tree(stores["envroot"]) == env_before, "the build touched the store the environment names"
    assert f"  Saved to: {written}" in out, out
    assert str(named_by_env) not in out, out


def test_the_legacy_index_note_also_names_the_store_that_is_written(stores, capsys):
    out = _build(stores, capsys, "GRIPSPACE_ROOT", legacy=Path("/legacy/index"))

    assert f"New location: {stores['written']}" in out, out
    assert str(stores["named_by_env"]) not in out, out


def test_control_without_an_environment_override_still_prints_the_written_store(stores, capsys, monkeypatch):
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    out = _build(stores, capsys, None)

    assert f"  Saved to: {stores['written']}" in out, out
