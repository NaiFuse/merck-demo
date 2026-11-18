"""Shared tender post-processing and DB writes for all tender connectors (TED, UK FTS).

Rule 7: save() is called only after all network work is done; each batch opens, writes and closes
its own connection.
"""

from __future__ import annotations

from pathlib import Path

from src.connectors import ecb_fx
from src.db import connect, init_db
from src.models.scope import tag_lab_water

WRITE_BATCH = 500


def apply_scope_flags(tenders: list[dict], awards: list[dict]) -> None:
    tmap = {t["notice_id"]: t for t in tenders}
    for a in awards:
        t = tmap[a["notice_id"]]
        others = [x for x in t.get("_lot_texts", []) if x != a.get("_own_text")]
        a["scope_note"] = tag_lab_water(a["winner_name"], t.get("title"), a.get("_own_text"), others)


def apply_fx(tenders: list[dict], awards: list[dict], rates: dict[tuple[str, str], float]) -> None:
    month = {t["notice_id"]: (t["award_date"] or t["publication_date"]).strftime("%Y-%m") for t in tenders}
    for t in tenders:
        t["value_eur"] = ecb_fx.to_eur(t["value"], t["currency"], month[t["notice_id"]], rates)
    for a in awards:
        a["value_eur"] = ecb_fx.to_eur(a["value"], a["currency"], month[a["notice_id"]], rates)


# ---------------------------------------------------------------- DB writes (short-lived connections)

TENDER_COLS = ["notice_id", "country", "buyer", "cpv", "value_eur", "award_date", "winner_name", "winner_id",
               "notice_type", "publication_date", "buyer_legal_type", "cpv_all", "value", "currency",
               "value_field", "n_winners", "source_url", "api_query", "retrieved_at", "raw_json",
               "is_framework", "title", "source"]
AWARD_COLS = ["notice_id", "seq", "winner_name", "winner_id", "value", "currency", "value_eur", "alignment", "grade",
              "evidence", "source_url", "scope_note"]


def _batches(rows: list, n: int):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def save(tenders: list[dict], awards: list[dict], fx: list[ecb_fx.FxRate],
         db_path: Path | str | None = None, batch_size: int = WRITE_BATCH) -> None:
    """Upsert in batches; each batch opens, writes and closes its own connection."""
    with connect(db_path) as con:
        init_db(con)
        if fx:
            con.executemany(
                "INSERT OR REPLACE INTO fx_rates VALUES (?, ?, ?, ?, ?)",
                [(r.currency, r.month, r.units_per_eur, r.source_url, r.retrieved_at) for r in fx],
            )
    by_notice: dict[str, list[dict]] = {}
    for a in awards:
        by_notice.setdefault(a["notice_id"], []).append(a)
    for batch in _batches(tenders, batch_size):
        ids = [t["notice_id"] for t in batch]
        with connect(db_path) as con:
            con.execute("BEGIN")
            con.executemany(
                f"INSERT OR REPLACE INTO tenders ({', '.join(TENDER_COLS)}) VALUES ({', '.join('?' * len(TENDER_COLS))})",
                [[t[c] for c in TENDER_COLS] for t in batch],
            )
            con.execute("DELETE FROM tender_awards WHERE list_contains(?, notice_id)", [ids])
            rows = [[a[c] for c in AWARD_COLS] for nid in ids for a in by_notice.get(nid, [])]
            if rows:
                con.executemany(f"INSERT INTO tender_awards ({', '.join(AWARD_COLS)}) VALUES ({', '.join('?' * len(AWARD_COLS))})", rows)
            con.execute("COMMIT")
