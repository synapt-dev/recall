"""A module this package ADDS carries no issue-style reference.

WHY THIS ROW EXISTS. The new ``vector_math`` module cited a same-repo issue
reference in a docstring, and that number resolves to a real, unrelated, CLOSED
issue -- so a public reader who followed it landed somewhere that had nothing to do
with the code. The reference was there only because the issue actually being
described is not public.

NO SCANNER CAN SEE THAT, WHICH IS THE POINT. The reference was well-formed, so a
leak scan was right to report nothing: the defect is not the SHAPE of the text, it
is that the number resolves to the wrong thing. A reader is the only instrument
that catches it, and this row is the mechanical half -- it makes the ABSENCE of a
reference checkable rather than remembered.

THIS FILE NAMES NO NUMBER EITHER, deliberately. The concrete number is exactly what
a reader would follow, so writing it here would reproduce the defect inside the
file that exists to prevent it. The shape is described; the digits are not.

SCOPE, deliberately narrow, and this is the part to preserve. The repository cites
issue numbers in code comments as an established convention -- hundreds of them on
dev -- so a package-wide ban would fail on text this change never touched and would
be switched off within a day. This row asserts the rule only for the module this
change ADDS, where the change is free to set the standard without a migration.
"""

from __future__ import annotations

import re
from pathlib import Path

MODULE = (
    Path(__file__).resolve().parents[2]
    / "src" / "synapt" / "recall" / "vector_math.py"
)

# An issue-style reference: a repo-ish slug, a hash, and a number.
ISSUE_REF = re.compile(r"[a-z][a-z-]*#[0-9]+")


def test_the_added_module_carries_no_issue_reference() -> None:
    hits = ISSUE_REF.findall(MODULE.read_text(encoding="utf-8"))
    assert hits == [], (
        "a same-repo issue reference in published code is unverifiable by a "
        f"public reader and was wrong here once: {hits}. State the property the "
        "reference was standing in for instead of citing a number."
    )


def test_the_check_can_actually_fire() -> None:
    """CONTROL, so the row above is not inert: the pattern must match the very
    text it was written for. A regex that matches nothing would make the assertion
    above pass for every possible input."""
    assert ISSUE_REF.findall("see repo#123 for why") == ["repo#123"]
    assert ISSUE_REF.findall("no reference here at all") == []
