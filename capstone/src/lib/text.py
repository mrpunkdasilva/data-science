"""Shared text helpers for cluster descriptors.

Kept small and dependency-free so the CLI, the dashboard and the report scripts
all share one definition of "top terms". A naive word count mixes in stopwords,
punctuation and dates (for example ``para``, ``res.`` or ``20.08.1998)``), which
reads badly on screen; this module keeps only meaningful alphabetic words.
"""

import re
from collections import Counter

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_STOPWORDS = frozenset(
    {
        "para",
        "pela",
        "pelo",
        "pelas",
        "pelos",
        "como",
        "mais",
        "menos",
        "sobre",
        "entre",
        "contra",
        "desde",
        "apos",
        "após",
        "ate",
        "até",
        "quando",
        "onde",
        "qual",
        "quais",
        "seja",
        "sera",
        "será",
        "sao",
        "são",
        "esta",
        "este",
        "estes",
        "estas",
        "essa",
        "esse",
        "essas",
        "esses",
        "isso",
        "aquilo",
        "aquele",
        "aquela",
        "dos",
        "das",
        "uma",
        "umas",
        "com",
        "sem",
        "sob",
        "que",
        "nao",
        "não",
        "por",
        "aos",
        "ainda",
        "tambem",
        "também",
        "entao",
        "então",
        "porem",
        "porém",
        "apenas",
        "assim",
        "cada",
        "mesmo",
        "toda",
        "todo",
        "todos",
        "todas",
    }
)


def top_terms(texts: list[str], n: int = 8) -> list[str]:
    """Return the ``n`` most frequent meaningful words across ``texts``.

    Punctuation and standalone numbers or dates are dropped, and a small
    Portuguese stopword list is removed, so the result reads as topic words
    rather than as raw token noise.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        for word in _WORD_RE.findall(str(text).lower()):
            if len(word) > 3 and word not in _STOPWORDS:
                counter[word] += 1
    return [word for word, _ in counter.most_common(n)]
