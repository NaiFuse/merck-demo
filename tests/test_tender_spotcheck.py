"""UK spot-check: deterministic picks and field comparison against a live-shaped FTS release (offline)."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.connectors import uk_fts
from src.models import tender_spotcheck as sc

PKG = json.loads((Path(__file__).parent / "fixtures" / "fts_release_min.json").read_text(encoding="utf-8"))


def _row(**kw):
    rel = next(r for r in PKG["releases"] if r["id"] == "030000-2025")
    t, awards = uk_fts.parse_release(rel, {a["id"] for a in rel["awards"]}, "u", pd.Timestamp("2026-11-18"),
                                     set(uk_fts.release_cpvs(rel)))
    a = awards[0]
    base = dict(notice_id=t["notice_id"], winner_name=a["winner_name"], value=a["value"], award_currency=a["currency"],
                evidence=a["evidence"], alignment=a["alignment"], buyer=t["buyer"], title=t["title"],
                publication_date=t["publication_date"], is_framework=t["is_framework"])
    return pd.Series(base | kw)


def test_compare_passes_on_matching_release():
    checks = sc.compare(_row(), PKG)
    assert checks and all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_compare_flags_value_and_winner_differences():
    bad = {c.field: c.ok for c in sc.compare(_row(value=1.0, winner_name="Someone Else"), PKG)}
    assert bad["value (orig. currency)"] is False and bad["winner"] is False


def test_pick_three_distinct_notices_with_rules():
    d = pd.DataFrame({
        "notice_id": ["FTS-1", "FTS-1", "FTS-2", "FTS-3", "FTS-4"],
        "value_eur": [9e6, 8e6, 5e6, 1e6, 2e6],
        "is_framework": [True, True, False, True, True],
        "group": ["other", "other", "other", "merck", "other"],
    })
    picks = sc.pick(d)
    assert [r["notice_id"] for _, r in picks] == ["FTS-1", "FTS-2", "FTS-3"]
    assert picks[2][0] == "largest UK award won by a peer"
