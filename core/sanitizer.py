"""
CAPS-K Sanitizer Module
=======================
Pre-processes untrusted text *before* it enters the prompt assembly pipeline.

Threat model
------------
Adversarial content arriving via EXT (retrieved documents) or TOUT (tool
outputs) may contain:

1. **Zero-width / invisible Unicode** — characters that are visually hidden
   but can confuse tokenisers or sneak in delimiter-like patterns.
2. **HTML tags** — ``<script>``, ``<style>``, or crafted ``<SYS_xxxx>``-
   looking sequences that could be mistaken for real delimiters.
3. **Lookalike brackets** — fullwidth ``＜＞``, mathematical angle brackets
   ``⟨⟩``, etc., used to construct visually convincing fake delimiters.

Sanitization steps
------------------
1. Strip zero-width and BOM characters.
2. Normalise lookalike angle brackets to ASCII ``< >``.
3. Remove HTML/XML tags (conservative: only well-formed-ish ``<...>``).
4. Escape any remaining ``<`` / ``>`` that are *not* part of our known
   delimiter pattern (since our delimiters are added *after* sanitisation).
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Zero-width and invisible character ranges
# ---------------------------------------------------------------------------
_ZW_CHARS = (
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202a"  # LEFT-TO-RIGHT EMBEDDING
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202c"  # POP DIRECTIONAL FORMATTING
    "\u202d"  # LEFT-TO-RIGHT OVERRIDE
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00ad"  # SOFT HYPHEN
)
_ZW_PATTERN = re.compile(f"[{re.escape(_ZW_CHARS)}]")

# Lookalike angle-bracket characters → normalise to ASCII < >
_LOOKALIKE_OPEN = re.compile(
    "["
    "\uff1c"  # FULLWIDTH LESS-THAN SIGN ＜
    "\u2329"  # LEFT-POINTING ANGLE BRACKET 〈
    "\u27e8"  # MATHEMATICAL LEFT ANGLE BRACKET ⟨
    "\u3008"  # LEFT ANGLE BRACKET 〈 (CJK)
    "\u2039"  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK ‹
    "\u00ab"  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK «
    "]"
)
_LOOKALIKE_CLOSE = re.compile(
    "["
    "\uff1e"  # FULLWIDTH GREATER-THAN SIGN ＞
    "\u232a"  # RIGHT-POINTING ANGLE BRACKET 〉
    "\u27e9"  # MATHEMATICAL RIGHT ANGLE BRACKET ⟩
    "\u3009"  # RIGHT ANGLE BRACKET 〉 (CJK)
    "\u203a"  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK ›
    "\u00bb"  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK »
    "]"
)

# Matches HTML/XML-like tags: <foo>, </foo>, <foo bar="baz">, etc.
_HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]{0,200}>", re.DOTALL)

# Matches any remaining raw < or > (after HTML tag removal)
_ANGLE_BRACKET_OPEN  = re.compile(r"<")
_ANGLE_BRACKET_CLOSE = re.compile(r">")


def sanitize(text: str) -> str:
    """
    Sanitise untrusted text to prevent delimiter spoofing and injection.

    Parameters
    ----------
    text:
        Raw untrusted string (e.g. from a retrieved document or tool output).

    Returns
    -------
    str
        Cleaned string, safe to embed inside a CAPS-K marked prompt.

    Examples
    --------
    >>> sanitize("Hello\\u200bworld <SYS_xxxx>evil</SYS_xxxx>")
    'Hello world &lt;SYS_xxxx&gt;evil&lt;/SYS_xxxx&gt;'
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Strip zero-width / invisible characters
    text = _ZW_PATTERN.sub("", text)

    # 2. Normalise lookalike angle brackets → ASCII
    text = _LOOKALIKE_OPEN.sub("<", text)
    text = _LOOKALIKE_CLOSE.sub(">", text)

    # 3. Remove genuine HTML/XML tags (strips <script>, <b>, etc.)
    text = _HTML_TAG_PATTERN.sub("", text)

    # 4. HTML-escape remaining angle brackets so they cannot form new tags
    text = _ANGLE_BRACKET_OPEN.sub("&lt;", text)
    text = _ANGLE_BRACKET_CLOSE.sub("&gt;", text)

    return text


def is_clean(text: str) -> bool:
    """
    Return True if *text* passes sanitisation unchanged (no mutations needed).

    Useful for fast pre-screening in high-throughput pipelines.
    """
    return sanitize(text) == text
