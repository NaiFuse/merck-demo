"""Scope flags for tender awards that are outside the DS-equivalent categories.

Lab water (Milli-Q systems and consumables) is excluded per CLAUDE.md. Only Merck awards are checked,
because Merck is the lab-water vendor whose value would otherwise inflate the DS-equivalent figure.
"""

from __future__ import annotations

import re

from src.models.winners import MERCK, _fold, winner_group

# Multilingual: EN, DE, FR, NL, SV/DA/NO, FI + Merck lab-water brands.
LAB_WATER_RE = re.compile(
    r"milli[\s-]?q|\belix\b|"
    r"lab(oratory)? water|ultra[\s-]?pure water|pure water|water purification|water system|"
    r"reinstwasser|reinwasser|wasseraufbereitung|laborwasser|"
    r"eau ultra[\s-]?pure|eau pure|purification d.eau|production d.eau|eau de laboratoire|"
    r"ultrapuur water|zuiver water|waterzuivering|"
    r"ultrarent vatten|renvatten|vattenrening|rent vand|vandrens|rent vann|vannrens|"
    r"puhdas vesi|vedenpuhdist"
)

LAB_WATER = "lab_water"
LAB_WATER_POSSIBLE = "lab_water_possible"


def is_lab_water(text: str | None) -> bool:
    return bool(text) and bool(LAB_WATER_RE.search(_fold(text)))


def tag_lab_water(winner_name: str | None, notice_title: str | None, own_lot_text: str | None,
                  other_lot_texts: list[str]) -> str | None:
    """'lab_water' when the notice title or the award's own lot text says lab water;
    'lab_water_possible' when only some other lot of the notice does (lot not attributable)."""
    if winner_group(winner_name) != MERCK:
        return None
    if is_lab_water(notice_title) or is_lab_water(own_lot_text):
        return LAB_WATER
    if any(is_lab_water(t) for t in other_lot_texts):
        return LAB_WATER_POSSIBLE
    return None
