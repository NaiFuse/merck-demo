"""Find numbers in a quote and check that one of them is a given value (rule 1: the quote must contain the number).

Handles US (1,234.5) and German (1.234,5) formats, ambiguous single separators (1,234 / 1.234 -> both readings),
space-grouped thousands as in the EU Official Journal (EUR 5 538 000 -> an extra reading),
negatives as -x or (x), percentages and scale words after the number (million, bn, Mio., Mrd., ...).
A bare number (no scale word) is read in the fact's own unit, as in a table headed "in millions".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# unit suffix -> multiplier (unit strings look like 'USD_m', 'EUR_k', 'EUR', '%', 'count')
UNIT_SCALE = {"": 1.0, "k": 1e3, "m": 1e6, "bn": 1e9}

SCALE_WORDS = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3, "tsd": 1e3, "tsd.": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "m": 1e6, "mio": 1e6, "mio.": 1e6, "millionen": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9, "mrd": 1e9, "mrd.": 1e9, "milliarden": 1e9,
}

_NUM_RE = re.compile(
    r"(?P<open>\()?"
    r"(?P<sign>[-−–])?\s?"
    r"(?:[$€£]|USD|EUR|GBP)?\s?"
    r"(?P<num>\d[\d.,]*\d|\d)"
    r"(?P<close>\))?"
    r"(?:\s?(?P<pct>%|percent|prozent))?"
    r"(?:\s?(?P<scale>[A-Za-z]+\.?))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Reading:
    value: float      # number as written, sign applied, before scale word
    decimals: int     # digits after the decimal separator (for rounding tolerance)
    scale: float      # multiplier from a scale word, 1.0 if none
    has_scale: bool
    pct: bool


def _readings_of(num: str) -> list[tuple[float, int]]:
    """All plausible (value, decimals) readings of a digit string with , and . separators."""
    has_c, has_d = "," in num, "." in num
    if not has_c and not has_d:
        return [(float(num), 0)]
    if has_c and has_d:
        dec = "," if num.rfind(",") > num.rfind(".") else "."
        thou = "." if dec == "," else ","
        intpart, frac = num.rsplit(dec, 1)
        if dec in intpart:
            return []
        return [(float(intpart.replace(thou, "") + "." + frac), len(frac))]
    sep = "," if has_c else "."
    parts = num.split(sep)
    if len(parts) > 2:  # 1,234,567 -> thousands separators only
        if all(len(p) == 3 for p in parts[1:]):
            return [(float("".join(parts)), 0)]
        return []
    intpart, frac = parts
    as_decimal = (float(intpart + "." + frac), len(frac))
    if len(frac) == 3:  # 1,234 or 1.234: thousands (US/DE) or decimal (DE/US)
        return [(float(intpart + frac), 0), as_decimal]
    return [as_decimal]


# Space-grouped thousands as in the EU Official Journal ('EUR 5 538 000'): regular, no-break, thin spaces.
_SPACED_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u2009\u202f]\d{3})+(?![\d.,]?\d)")


def find_numbers(text: str) -> list[Reading]:
    out: list[Reading] = []
    for m in _NUM_RE.finditer(text or ""):
        neg = bool(m.group("sign")) or bool(m.group("open") and m.group("close"))
        scale_word = (m.group("scale") or "").lower()
        scale = SCALE_WORDS.get(scale_word)
        for val, dec in _readings_of(m.group("num")):
            out.append(Reading(-val if neg else val, dec, scale or 1.0, scale is not None, bool(m.group("pct"))))
    for m in _SPACED_RE.finditer(text or ""):   # extra reading only; the separate groups above are kept
        out.append(Reading(float(re.sub(r"\D", "", m.group(0))), 0, 1.0, False, False))
    return out


def unit_multiplier(unit: str) -> float:
    """'USD_m' -> 1e6, 'EUR_k' -> 1e3, 'EUR' -> 1, '%' -> 1."""
    if "_" not in unit:
        return 1.0
    suffix = unit.rsplit("_", 1)[1].lower()
    if suffix not in UNIT_SCALE:
        raise ValueError(f"unknown unit suffix in {unit!r}")
    return UNIT_SCALE[suffix]


def quote_contains_value(quote: str, value: float, unit: str, rel_tol: float = 1e-9) -> bool:
    """True if some number in the quote equals value (in unit) within its displayed rounding.

    '$10.9 billion' matches 10_898 USD_m (shown to 1 decimal of a billion -> +-50 USD_m).
    A bare number is read in the fact's unit. Percent facts need a % reading or a bare number.
    """
    mult = unit_multiplier(unit)
    is_pct_unit = unit.strip() in ("%", "pct", "percent")
    for r in find_numbers(quote):
        if is_pct_unit and r.has_scale:
            continue
        if r.has_scale:
            in_unit = r.value * r.scale / mult
            half_step = 0.5 * 10 ** (-r.decimals) * r.scale / mult
        else:
            in_unit = r.value
            half_step = 0.5 * 10 ** (-r.decimals)
        for v in (value, -value):
            if abs(in_unit - v) <= half_step + rel_tol * max(abs(v), 1.0):
                return True
    return False
