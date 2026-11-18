"""Academia buyer filter for all tender sources (rules in config/academia_buyers.yaml).

Keeps universities, university purchasing consortia, (university) hospitals and hospital purchasing bodies,
hospital-running health regions, public research organisations and government labs. Excludes schools and school
consortia, utilities, municipalities, general purchasing bodies and unclassified buyers. Buyer name rules first,
the buyer-type field (FTS classification, TED legal type) as fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.models.winners import _fold
from src.settings import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "config" / "academia_buyers.yaml"
UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class BuyerClass:
    category: str
    keep: bool
    rule: str          # 'name:<pattern>' | 'type:<value>' | 'none'


@lru_cache
def _rules(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    comp = {sec: [(cat, re.compile(p)) for cat, ps in cfg[sec].items() for p in ps]
            for sec in ("exclude_first", "keep", "exclude")}
    return {"compiled": comp, "type_fallback": cfg["type_fallback"], "keep": set(cfg["keep_categories"])}


def _norm(s: str) -> str:
    return _fold(s).replace("æ", "ae").replace("ø", "o").replace("å", "a").replace("ð", "d").replace("þ", "th")


def classify_part(name: str, legal_type: str | None, source: str) -> BuyerClass:
    r = _rules()
    n = _norm(name)
    for sec in ("exclude_first", "keep", "exclude"):
        for cat, rx in r["compiled"][sec]:
            if rx.search(n):
                return BuyerClass(cat, cat in r["keep"], f"name:{rx.pattern[:40]}")
    lt = legal_type if isinstance(legal_type, str) else ""
    for key, cat in (r["type_fallback"].get(source) or {}).items():
        hit = (key in lt) if key.startswith(";") else (lt == key or lt.split(";")[0] == key)
        if hit:
            return BuyerClass(cat, cat in r["keep"], f"type:{key}")
    return BuyerClass(UNCLASSIFIED, False, "none")


def classify_buyer(name: str | None, legal_type: str | None = None, source: str = "ted") -> BuyerClass:
    """Multi-buyer names ('A; B') are kept if any part is kept (category of the first kept part)."""
    if not name or not str(name).strip():
        return BuyerClass(UNCLASSIFIED, False, "no buyer name")
    parts = [p for p in re.split(r";\s*", str(name)) if p.strip()]
    classes = [classify_part(p, legal_type, source) for p in parts]
    return next((c for c in classes if c.keep), classes[0])
