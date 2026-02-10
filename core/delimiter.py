"""
CAPS-K Delimiter Module
=======================
Generates per-request randomised prefix-only markers for each authority
category, and implements K-token interleaving — the core provenance
signalling mechanism.

Design (v2 — clean)
--------------------
* Prefix-only markers: <CAT_xxxx>  (NO closing tags)
* A single marker is re-inserted every K whitespace tokens
* No adjacent resets — one marker per segment boundary, no duplicates
* Per-request unique suffixes make delimiter spoofing from within EXT content
  practically impossible (attacker cannot know the session suffix)

Example output for K=6, EXT:
    <EXT_abc7> word0 word1 word2 word3 word4 word5
    <EXT_abc7> word6 word7 ... word11
    <EXT_abc7> word12 ...

The model sees consistent provenance anchors without noisy open/close clutter.
"""

import random
import string
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CATEGORIES, DELIMITER_SUFFIX_LEN


def _random_suffix(length: int = DELIMITER_SUFFIX_LEN) -> str:
    """Return a random lowercase hex string of *length* characters."""
    return "".join(random.choices(string.hexdigits[:16], k=length))


def generate_delimiters(categories: list[str] | None = None) -> dict[str, str]:
    """
    Generate a per-request set of randomised prefix-only markers.

    Each category gets a unique suffix to prevent cross-category spoofing.

    Parameters
    ----------
    categories:
        List of category names. Defaults to ``config.CATEGORIES``.

    Returns
    -------
    dict mapping category name -> prefix marker string, e.g.::

        {
            "SYS":  "<SYS_a3f2>",
            "USR":  "<USR_9b1c>",
            "EXT":  "<EXT_7d44>",
            ...
        }
    """
    if categories is None:
        categories = CATEGORIES

    delimiters: dict[str, str] = {}
    used_suffixes: set[str] = set()

    for cat in categories:
        suffix = _random_suffix()
        while suffix in used_suffixes:
            suffix = _random_suffix()
        used_suffixes.add(suffix)
        delimiters[cat] = f"<{cat}_{suffix}>"

    return delimiters


def mark_every_k_tokens(
    text: str,
    marker: str,
    k: int,
) -> str:
    """
    Insert a prefix marker at the start and then every *k* whitespace tokens.

    Output structure::

        <CAT_xxxx> tok0 tok1 … tok(k-1)
        <CAT_xxxx> tokk … tok(2k-1)
        <CAT_xxxx> tok(2k) …

    One marker per segment — no closing tags, no adjacent duplicates.

    Parameters
    ----------
    text:
        Raw text to mark (already sanitised).
    marker:
        Prefix marker string, e.g. ``<EXT_abc7>``.
    k:
        Number of tokens per segment. Must be >= 1.

    Returns
    -------
    str
        Text with interleaved prefix markers.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    tokens = text.split()

    if not tokens:
        return marker

    segments: list[str] = []
    for i in range(0, len(tokens), k):
        chunk = " ".join(tokens[i : i + k])
        segments.append(f"{marker} {chunk}")

    return "\n".join(segments)
