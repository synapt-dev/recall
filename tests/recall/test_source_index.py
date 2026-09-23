from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from synapt.recall import source_index
from synapt.recall.source_index import (
    SOURCE_INDEX_SUPPORTED,
    DescriptorSourceAdapter,
    SourceAdmission,
    SourceLimits,
    compose_source_results,
    parse_markdown,
    render_source_results,
    search_source,
    sync_source,
)


pytestmark = pytest.mark.skipif(
    not SOURCE_INDEX_SUPPORTED,
    reason="descriptor adapter is POSIX-only: os.O_DIRECTORY unavailable",
)


def _admission(root_fd: int, *, disclosure: str = "hidden") -> SourceAdmission:
    return SourceAdmission(
        source_id="source-opaque-1",
        scope_capability=b"capability",
        root_handle=root_fd,
        root_handle_id="root-handle-opaque-1",
        admission_epoch=1,
        policy_epoch=1,
        path_disclosure=disclosure,
    )


def _opener(path: Path, calls: list[str] | None = None):
    def open_store() -> sqlite3.Connection:
        if calls is not None:
            calls.append("open")
        return sqlite3.connect(path)

    return open_store


def test_memory_file_flows_through_distinct_source_index_and_render(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "index.md").write_text(
        "# Decisions\n\nScope refraction keeps projection boundaries visible.\n",
        encoding="utf-8",
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        admission = _admission(root_fd)
        db_path = tmp_path / "private-source.db"
        receipt = sync_source(
            admission,
            DescriptorSourceAdapter(),
            _opener(db_path),
            lambda candidate: candidate.scope_capability == b"capability",
        )
        assert receipt.state == "complete"
        assert receipt.documents_seen == 1
        assert receipt.units_published == 1

        results = search_source(
            admission,
            _opener(db_path),
            lambda candidate: candidate.scope_capability == b"capability",
            "scope refraction",
        )
    finally:
        os.close(root_fd)

    assert len(results) == 1
    result = results[0]
    assert result.source_kind == "memory_file"
    assert result.structural_address == "Decisions [1]"
    assert result.lifecycle == "current"
    assert result.relative_path is None
    assert "scope refraction" in result.content.lower()
    rendered = render_source_results(results)
    assert "[source:memory_file · current · Decisions [1] · revision " in rendered
    assert str(root) not in rendered
    assert "document_sha256" not in rendered
    assert compose_source_results("conversation result", results).startswith(
        "conversation result\n\n[source:"
    )

    connection = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()
    assert "source_units" in tables
    assert not ({"chunks", "knowledge", "query_tail_chunks"} & tables)


def test_unchanged_second_scan_reuses_document_without_parsing(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "rule.md").write_text(
        "# Rule\n\nPrefer the narrow seam.\n", encoding="utf-8"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    parser_calls: list[bytes] = []

    def counting_parser(content: bytes):
        parser_calls.append(content)
        return parse_markdown(content)

    try:
        admission = _admission(root_fd)
        opener = _opener(tmp_path / "private-source.db")
        first = sync_source(
            admission,
            DescriptorSourceAdapter(),
            opener,
            lambda _candidate: True,
            parser=counting_parser,
        )
        second = sync_source(
            admission,
            DescriptorSourceAdapter(),
            opener,
            lambda _candidate: True,
            parser=counting_parser,
        )
    finally:
        os.close(root_fd)

    assert first.state == second.state == "complete"
    assert len(parser_calls) == 1
    assert second.documents_reused == 1
    assert second.units_published == 0


def test_parser_limit_receipt_reports_attempted_units_and_parser_units_knob(
    tmp_path: Path,
) -> None:
    """An aggregate parser refusal tells the operator what was attempted and
    exactly which limit to inspect, without changing atomic publication."""
    root = tmp_path / "memory"
    root.mkdir()
    (root / "oversized.md").write_text(
        "# First unit\n\nOne.\n\n# Second unit\n\nTwo.\n", encoding="utf-8"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt = sync_source(
            _admission(root_fd),
            DescriptorSourceAdapter(),
            _opener(tmp_path / "private-source.db"),
            lambda _candidate: True,
            limits=SourceLimits(parser_units=1),
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "parser_limit_exceeded"
    assert receipt.units_attempted == 2
    assert receipt.parser_units == 1


def test_unauthorized_calls_open_nothing_and_return_no_corpus_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    store_calls: list[str] = []
    try:
        admission = _admission(root_fd)
        opener = _opener(tmp_path / "must-not-exist.db", store_calls)
        receipt = sync_source(
            admission,
            DescriptorSourceAdapter(),
            opener,
            lambda _candidate: False,
        )
        results = search_source(
            admission,
            opener,
            lambda _candidate: False,
            "anything",
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "unauthorized"
    assert receipt.generation is None
    assert receipt.documents_seen is None
    assert results == []
    assert store_calls == []
    assert not (tmp_path / "must-not-exist.db").exists()


def test_enumerate_refuses_cleanly_when_platform_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SOURCE_INDEX_SUPPORTED False must raise a clear _ScanFailure("unsupported")
    before any os.O_DIRECTORY use -- not let its absence surface as a raw
    AttributeError deep in the walk, which is the actual failure this
    platform ever produces on Windows.

    Setup happens with the real O_DIRECTORY still present (this test only
    runs where it exists, per the module skip); the attribute is deleted
    only after the admission fd is built, so removing the guard below (as a
    manual mutation check) makes enumerate() reach the deleted attribute and
    raise a genuine AttributeError -- proving this test is bound to the
    guard, not merely to the derived flag."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "child").mkdir()
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(source_index, "SOURCE_INDEX_SUPPORTED", False)
        monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
        admission = _admission(root_fd)
        with pytest.raises(source_index._ScanFailure) as exc_info:
            list(DescriptorSourceAdapter().enumerate(admission, SourceLimits()))
        assert exc_info.value.state == "unsupported"
    finally:
        os.close(root_fd)


# --- parser_units default vs a measured coordinator-scale input (recall#1197) ---

# The measured representative input: a coordinator-scale memory directory that
# parses into 1,730 units across 691 files. The ceiling is on AGGREGATE UNITS,
# so the fixture below matches the aggregate and not the file count -- the
# refusal counts parsed units (`total_units > limits.parser_units`), never
# files, and 691 tiny files would only make the test slower, not more faithful.
_COORDINATOR_SCALE_UNITS = 1_730
_FILES = 173          # x 10 headings each = the measured aggregate


def _coordinator_scale_fixture(root: Path) -> int:
    """Write the fixture and return the unit count its own parser reports.

    The count is DERIVED from parse_markdown rather than assumed from the
    headings I wrote, so a change to the parser shows up as a different
    fixture rather than as a silently weaker test.
    """
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(_FILES):
        body = "".join(
            f"# Section {i}.{j}\n\nBody text for unit {i}.{j}.\n\n"
            for j in range(1, 11)
        )
        (root / f"mem-{i:03d}.md").write_text(body, encoding="utf-8")
        total += len(parse_markdown(body.encode("utf-8")))
    return total


def test_default_parser_units_admits_the_measured_coordinator_scale_input(
    tmp_path: Path,
) -> None:
    """The DEFAULT ceiling must admit the measured representative input.

    recall#1197 measured the default of 1,000 refusing a real memory directory
    of 1,730 units with a `parser_limit_exceeded` receipt and no generation.
    This test passes NO limits, so it exercises the default a user gets.
    """
    root = tmp_path / "memory"
    units = _coordinator_scale_fixture(root)
    assert units == _COORDINATOR_SCALE_UNITS, (
        f"fixture drift: parser reports {units} units, not the measured "
        f"{_COORDINATOR_SCALE_UNITS}"
    )

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt = sync_source(
            _admission(root_fd),
            DescriptorSourceAdapter(),
            _opener(tmp_path / "source.db"),
            lambda _candidate: True,
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "complete", (
        f"the default ceiling refused the measured input: state="
        f"{receipt.state!r} units_attempted={receipt.units_attempted} "
        f"parser_units={receipt.parser_units}"
    )
    assert receipt.units_published == _COORDINATOR_SCALE_UNITS
    assert receipt.generation == 1


def test_default_parser_units_is_the_ratified_ceiling() -> None:
    """The default is the ratified number, not merely "bigger than 1,730".

    A default that admitted today's fixture by luck would pass the test above
    and drift on the next one; the ratified policy is 5,000.
    """
    assert SourceLimits().parser_units == 5_000


def test_parser_limit_refusal_stays_atomic_under_an_explicit_low_limit(
    tmp_path: Path,
) -> None:
    """Raising the default must not weaken the refusal itself.

    With a limit set below the fixture, the refusal must still be atomic: a
    `parser_limit_exceeded` receipt, the attempted count and the knob reported,
    and NOTHING published -- no generation, no queryable documents. A partial
    publish here would be worse than the refusal it replaced.
    """
    root = tmp_path / "memory"
    _coordinator_scale_fixture(root)
    store = tmp_path / "source.db"

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    admission = _admission(root_fd)
    try:
        receipt = sync_source(
            admission,
            DescriptorSourceAdapter(),
            _opener(store),
            lambda _candidate: True,
            limits=SourceLimits(parser_units=100),
        )
        results = search_source(
            admission, _opener(store), lambda _candidate: True, "Body text"
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "parser_limit_exceeded"
    assert receipt.parser_units == 100
    assert receipt.units_attempted is not None and receipt.units_attempted > 100
    assert receipt.generation is None
    assert receipt.units_published is None
    assert results == [], "a refused scan left queryable documents behind"


def test_env_override_raises_or_lowers_the_ceiling_the_default_scan_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's override is read on the no-limits path.

    Set BELOW the fixture so the override is visible as a refusal rather than
    inferred from a passing scan, and assert the receipt names the overridden
    number -- the refusal has to report the ceiling that actually applied.
    """
    root = tmp_path / "memory"
    _coordinator_scale_fixture(root)
    monkeypatch.setenv("SYNAPT_SOURCE_PARSER_UNITS", "100")

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt = sync_source(
            _admission(root_fd),
            DescriptorSourceAdapter(),
            _opener(tmp_path / "source.db"),
            lambda _candidate: True,
        )
    finally:
        os.close(root_fd)

    assert receipt.state == "parser_limit_exceeded", (
        "the env override was ignored: the scan used the dataclass default"
    )
    assert receipt.parser_units == 100


def test_malformed_env_override_warns_and_falls_back_to_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A malformed override must not vanish silently, and must not be read as 0.

    Falls back to the ratified default and prints a warning naming the bad
    value, mirroring core._int_env_override's contract for the transcript
    ceilings.
    """
    root = tmp_path / "memory"
    root.mkdir()
    (root / "one.md").write_text("# Only unit\n\nBody.\n", encoding="utf-8")
    monkeypatch.setenv("SYNAPT_SOURCE_PARSER_UNITS", "banana")

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt = sync_source(
            _admission(root_fd),
            DescriptorSourceAdapter(),
            _opener(tmp_path / "source.db"),
            lambda _candidate: True,
        )
    finally:
        os.close(root_fd)

    err = capsys.readouterr().err
    assert receipt.state == "complete", "a malformed override changed the scan"
    assert "banana" in err and "not a valid integer" in err


def test_non_positive_env_override_is_not_read_as_an_unusable_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`0` must not mean "refuse every corpus, silently".

    A non-positive value fails `SourceLimits` construction, and the startup
    caller wraps the whole admission in ``except Exception: pass`` -- so
    without this fallback the source index would simply never publish, with
    nothing on stderr to say why. The scan must proceed at the ratified
    default and name the rejected value.
    """
    root = tmp_path / "memory"
    root.mkdir()
    (root / "one.md").write_text("# Only unit\n\nBody.\n", encoding="utf-8")
    monkeypatch.setenv("SYNAPT_SOURCE_PARSER_UNITS", "0")

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt = sync_source(
            _admission(root_fd),
            DescriptorSourceAdapter(),
            _opener(tmp_path / "source.db"),
            lambda _candidate: True,
        )
    finally:
        os.close(root_fd)

    err = capsys.readouterr().err
    assert receipt.state == "complete", (
        "a non-positive override changed or blocked the scan instead of "
        "falling back"
    )
    assert "SYNAPT_SOURCE_PARSER_UNITS=0" in err and "not a usable ceiling" in err
