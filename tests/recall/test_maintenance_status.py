"""The build free-memory floor as a config key, and `maintenance status`.

The floor is a HOST property: it reads env > global config > default and NEVER the project
layer,
because the project file is found from the caller's cwd and a project config committed to a
repository could lower a safety floor for everyone who clones it.

Every case here drives the REAL gate or the REAL command against a scratch HOME. Nothing
re-implements the decision, and no case asserts an exit path the caller never takes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from synapt.recall import cli
from synapt.recall.config import clear_config_cache

CONSTANT = cli.BUILD_MIN_FREE_INACTIVE_GB
FLOOR = cli.BUILD_MIN_FREE_HARD_FLOOR_GB


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch HOME carrying its own global config, so nothing shared is touched."""
    h = tmp_path / "home"
    (h / ".synapt").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("SYNAPT_BUILD_MIN_FREE_GB", raising=False)
    clear_config_cache()
    yield h
    clear_config_cache()


def write_global(home: Path, memory) -> None:
    (home / ".synapt" / "config.json").write_text(json.dumps({"memory": memory}))
    clear_config_cache()


def verdict(free_gb: float, monkeypatch) -> str:
    """The REAL gate's verdict at a pinned reading."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", f"0:0:{free_gb}:0")
    return cli._host_memory_verdict()[0]


def status_line(capsys) -> str:
    cli.cmd_maintenance(argparse.Namespace(command="maintenance", maintenance_action="status"))
    return capsys.readouterr().out


class TestTheKeyMovesTheRealGate:
    def test_defer_to_ok_and_back(self, home, monkeypatch, capsys):
        """The fruit, at the gate: the same host reading, two settings, two verdicts."""
        assert verdict(2.2, monkeypatch) == "refuse", "premise: unset, 2.2 GB is under the 6 GB default"

        write_global(home, {"build_min_free_gb": 2.0})
        assert verdict(2.2, monkeypatch) == "pass", "the key did not open the gate"
        out = status_line(capsys)
        assert "-> OK" in out and "source: global config" in out

        (home / ".synapt" / "config.json").unlink()
        clear_config_cache()
        assert verdict(2.2, monkeypatch) == "refuse", "the gate did not close again"
        out = status_line(capsys)
        assert "-> DEFER" in out and "source: default" in out

    def test_env_still_beats_the_global_config(self, home, monkeypatch):
        write_global(home, {"build_min_free_gb": 9.0})
        monkeypatch.setenv("SYNAPT_BUILD_MIN_FREE_GB", "2.0")
        assert verdict(2.2, monkeypatch) == "pass"
        assert cli._resolve_build_min_free_gb()[1] == "env"

    # The refuse LINE naming its source is witnessed by fruit on a real `catchup` rather than
    # here: asserting the text exists in the source would prove the string is written, not
    # that the line runs, and the line only runs inside a refused catchup.


class TestTheHardFloorAndBadTypes:
    def test_a_value_under_the_hard_floor_clamps_and_says_so(self, home, monkeypatch):
        write_global(home, {"build_min_free_gb": 0.1})
        value, source, note = cli._resolve_build_min_free_gb()
        assert value == FLOOR, "a setting must not be able to go under the hard floor"
        assert source == "global config"
        assert note == f"requested 0.1, using {FLOOR:g} (hard floor)", note

    @pytest.mark.parametrize(
        "raw,label",
        [(True, "bool"), (None, "null"), ([1, 2], "list"), ({"a": 1}, "object")],
    )
    def test_a_value_that_is_not_a_number_falls_back_AND_says_so(self, home, raw, label):
        """A boolean is here on purpose: float(True) is 1.0, so without the explicit guard a
        typo would become a 1 GB floor and read as configured."""
        write_global(home, {"build_min_free_gb": raw})
        value, source, note = cli._resolve_build_min_free_gb()
        assert value == CONSTANT, f"{label} produced floor={value!r}"
        assert source == "default"
        assert note.startswith("requested ") and note.endswith("(not a number)"), note

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", 0, -1])
    def test_a_number_the_gate_cannot_use_falls_back_and_says_so(self, home, raw):
        write_global(home, {"build_min_free_gb": raw})
        value, source, note = cli._resolve_build_min_free_gb()
        assert value == CONSTANT, f"{raw!r} produced floor={value!r}"
        assert note.endswith("(not a usable number)"), note

    def test_a_numeric_string_is_accepted_because_env_values_are_strings(self, home):
        """Deliberate: the env layer is always a string, so rejecting strings here would make
        the two layers disagree about the same text."""
        write_global(home, {"build_min_free_gb": "2.0"})
        value, source, note = cli._resolve_build_min_free_gb()
        assert (value, source, note) == (2.0, "global config", "")

    def test_an_absent_key_carries_no_note(self, home):
        """The absence of a setting must not be reported as a rejected one."""
        assert cli._resolve_build_min_free_gb() == (CONSTANT, "default", "")


class TestTheProjectLayerIsNotConsulted:
    def test_a_project_key_is_ignored_and_the_status_line_says_so(self, tmp_path, home, monkeypatch, capsys):
        project = tmp_path / "proj"
        (project / ".synapt" / "recall").mkdir(parents=True)
        (project / ".synapt" / "recall" / "config.json").write_text(
            json.dumps({"memory": {"build_min_free_gb": 1.0}})
        )
        monkeypatch.chdir(project)
        clear_config_cache()

        value, source, _ = cli._resolve_build_min_free_gb()
        assert (value, source) == (CONSTANT, "default"), (
            "a project config lowered a host guard; the project layer is never read "
            "for memory.*, because a committed project file would lower everyone's floor"
        )
        out = status_line(capsys)
        assert "IGNORED" in out and "memory.build_min_free_gb" in out, out
