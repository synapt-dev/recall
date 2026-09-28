"""Vector math shared by every similarity path in the package.

ONE cosine, ONE width rule, ONE place to change it. There were three copies of
this function before -- ``source_index._cosine``, ``core._cosine`` and
``embeddings.cosine_similarity`` -- and all three truncated the dot product at
``zip`` while normalising over the full vectors, so a pair of different widths
produced a plausible-looking number that was simply wrong.

WHY THIS MODULE IMPORTS NOTHING BUT ``math``: the modules that need a cosine sit
on hot import paths (``core`` is on the CLI cold-start path), and
``embeddings`` -- which used to own this function -- drags in ``urllib.request``,
which is why four of its callers import it lazily. A cosine needs no I/O and no
provider, so it lives here and costs nothing to import. ``embeddings``
re-exports it so its existing callers keep working and inherit the fix.
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, which must be the SAME width.

    A width mismatch RAISES rather than scoring. It used to score: the dot
    product below sums over ``zip(a, b)``, which stops at the shorter vector,
    while both norms sum over the FULL vectors -- so the numerator was truncated
    and the denominator was not, and a mismatched pair returned a plausible
    number with no meaning. Measured on the source index: a 2-wide row whose true
    similarity to the query is 0.8 scores 0.686 against a 3-wide query, and a
    wider query deflates further until rows fall under a similarity floor and
    drop out of the results with no error anywhere.

    Refusing is the invariant the store's write side already holds: a provider
    whose width is not the store's must not produce a blob, so a width mismatch
    must not produce a VALUE here either. Callers decide what to do about it --
    ``search_source`` skips the row and counts it, because a
    stored width is a property of when that row was written and refusing the
    whole query would take down the rows that ARE comparable.
    """
    if len(a) != len(b):
        raise ValueError(
            f"embedding width mismatch: {len(a)} != {len(b)} -- refusing to score "
            "vectors of different widths"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
