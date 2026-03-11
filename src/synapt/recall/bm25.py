"""Standalone Okapi BM25 implementation with zero external dependencies.

Provides keyword-based document retrieval using the BM25 scoring function.
Internalized from Synapt's api_index.py for standalone use.

Includes a lightweight suffix stripper for morphological normalization
(orphaned → orphan, deciding → decid, fixes → fix).
"""

from __future__ import annotations

import math
import re
from typing import List


# ---------------------------------------------------------------------------
# Suffix stripper for search recall
# ---------------------------------------------------------------------------
# A simple suffix-stripping stemmer optimized for search matching, not
# linguistic correctness.  Handles common English inflectional and
# derivational suffixes so morphological variants (orphan/orphaned/
# orphaning) map to the same token.
#
# The FTS5 path uses SQLite's built-in Porter tokenizer for more
# accurate stemming.  This stemmer covers the BM25 fallback path.


def _stem(word: str) -> str:
    """Strip common English suffixes to normalize morphological variants.

    Returns a stemmed form that groups inflected/derived words together.
    Not linguistically perfect — prioritizes search recall (matching
    'orphan' to 'orphaned') over precision.
    """
    if len(word) <= 3:
        return word

    # Compound derivational suffixes (longest first)
    for suffix in (
        "ingness", "fulness", "ousness", "iveness",
        "ization", "isation", "ational",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]

    for suffix in ("ement", "eness", "ments", "ating", "izing"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]

    # Simple derivational suffixes
    # -ment/-ness need stem >= 5 to avoid "comment" → "com" or "harness" → "har"
    for suffix in ("ness", "ment"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 5:
            return word[: -len(suffix)]
    for suffix in ("able", "ible"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]

    # Inflectional suffixes — order matters.
    # Step 1: strip plural -s/-es FIRST, then re-apply rules to the
    # de-pluralized form. This ensures "prefers" and "prefer" stem to
    # the same root ("pref"), fixing inconsistency where only one rule
    # fires and different inflections produce different stems.
    depluralized = word
    if word.endswith("sses"):
        depluralized = word[:-2]  # caresses → caress
    elif word.endswith("ies") and len(word) > 4:
        depluralized = word[:-2]  # ponies → poni
    elif word.endswith("es") and len(word) > 4:
        depluralized = word[:-2]  # fixes → fix
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        depluralized = word[:-1]  # decisions → decision

    # Step 2: apply derivational/other inflectional rules to the
    # (possibly de-pluralized) form.
    w = depluralized
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]  # orphaning → orphan, fixing → fix
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]  # orphaned → orphan, fixed → fix
    if w.endswith("ly") and len(w) > 4:
        return w[:-2]  # actually → actual
    if w.endswith("er") and len(w) > 5:
        return w[:-2]  # higher → high, prefer → pref, prefers → pref
    return depluralized


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization with suffix stripping.

    Only stems pure-alphabetic tokens (English words). Tokens containing
    dots, underscores, or digits are code identifiers / file paths and
    are kept as-is to avoid mangling ``api_index.py`` → ``api_index.pi``.
    """
    tokens = re.sub(r"[^a-zA-Z0-9_.]", " ", text.lower()).split()
    result = []
    for t in tokens:
        if len(t) <= 1:
            continue
        if t.isalpha():
            result.append(_stem(t))
        else:
            result.append(t)
    return result


class BM25:
    """Lightweight BM25 index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.df: dict[str, int] = {}  # document frequency per term
        self.idf: dict[str, float] = {}

    def index(self, documents: List[List[str]]) -> None:
        self.corpus = documents
        self.n_docs = len(documents)
        self.doc_len = [len(d) for d in documents]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)

        self.df = {}
        for doc in documents:
            seen = set(doc)
            for term in seen:
                self.df[term] = self.df.get(term, 0) + 1

        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        for q in query_tokens:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for i, doc in enumerate(self.corpus):
                tf = doc.count(q)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * numerator / denominator
        return scores
