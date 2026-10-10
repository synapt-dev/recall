"""The member repos a gripspace root DECLARES.

A gripspace root lists its members in ``.gitgrip/spaces/main/gripspace.yml``
(``repos:`` -> one key per member, each with a ``path:``). Reading that list is
declaring, not inferring: walking the root for ``.git`` directories also finds
review clones and scratch checkouts that are not members. Reference repos
(``reference: true``, read-only comparisons) are not members of the work.

The manifest is read with the standard library only: ``repos:`` entries sit at
two spaces of indent and their scalar fields at four, which is the one shape the
tooling writes. A file that does not have that shape yields no members, and the
caller falls back to its own refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MANIFEST_RELPATH = Path(".gitgrip") / "spaces" / "main" / "gripspace.yml"


@dataclass(frozen=True)
class Member:
    key: str          # the manifest key
    rel_path: str     # path under the root, without a leading "./"
    path: Path        # absolute
    cloned: bool      # has a .git


def _scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


def declared_members(root: Path) -> list[Member] | None:
    """The non-reference members the root's manifest declares, or ``None``
    when the root has no readable manifest or it declares no repos.

    A member whose directory is absent or has no ``.git`` is returned with
    ``cloned=False`` so the caller can say it was not searched.
    """
    manifest = Path(root) / MANIFEST_RELPATH
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return None

    entries: dict[str, dict[str, str]] = {}
    in_repos = False
    current: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0:
            in_repos = line == "repos:"
            current = None
            continue
        if not in_repos:
            continue
        if indent == 2 and line.endswith(":"):
            current = line[:-1].strip().strip("'\"")
            entries[current] = {}
        elif indent == 4 and current is not None and ":" in line:
            key, _, value = line.partition(":")
            entries[current][key.strip()] = _scalar(value)

    members: list[Member] = []
    for key, fields in entries.items():
        if fields.get("reference", "").lower() == "true":
            continue
        rel = fields.get("path", "")
        if not rel:
            continue
        rel = rel[2:] if rel.startswith("./") else rel
        absolute = (Path(root) / rel).resolve()
        members.append(Member(key, rel, absolute, (absolute / ".git").exists()))
    return members or None
