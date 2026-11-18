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


PLACEHOLDER_NOTE = " [negative placeholder value (e.g. TED -1 = not disclosed) stored as NULL]"


def null_placeholder_values(tenders: list[dict], awards: list[dict]) -> int:
    """Negative amounts are 'not disclosed' placeholders (TED uses -1): store NULL, keep the raw value in the
    evidence/value_field text and raw_json. Awards become alignment 'no_value', grade C. Returns rows changed."""
    n = 0
    for t in tenders:
        if t.get("value") is not None and t["value"] < 0:
            t["value_field"] = f"{t.get('value_field') or ''} raw={t['value']}{PLACEHOLDER_NOTE}".strip()
            t["value"] = None
            n += 1
    for a in awards:
        if a.get("value") is not None and a["value"] < 0:
            a["evidence"] = f"{a.get('evidence') or ''}{PLACEHOLDER_NOTE}"
            a["value"], a["alignment"], a["grade"] = None, "no_value", "C"
            n += 1
    return n


def migrate_placeholders(db_path: Path | str | None = None) -> dict:
    """One-off fix for rows stored before null_placeholder_values existed."""
    with connect(db_path) as con:
        init_db(con)
        con.execute("BEGIN")
        t = con.execute("SELECT count(*) FROM tenders WHERE value < 0 OR value_eur < 0").fetchone()[0]
        a = con.execute("SELECT count(*) FROM tender_awards WHERE value < 0 OR value_eur < 0").fetchone()[0]
        con.execute("UPDATE tenders SET value_field = trim(coalesce(value_field, '') || ' raw=' || value || ?), "
                    "value = NULL, value_eur = NULL WHERE value < 0 OR value_eur < 0", [PLACEHOLDER_NOTE])
        con.execute("UPDATE tender_awards SET evidence = coalesce(evidence, '') || ?, value = NULL, value_eur = NULL, "
                    "alignment = 'no_value', grade = 'C' WHERE value < 0 OR value_eur < 0", [PLACEHOLDER_NOTE])
        con.execute("COMMIT")
    return {"tenders": t, "awards": a}


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
    null_placeholder_values(tenders, awards)
    for t in tenders:   # recompute EUR after nulling (apply_fx may have converted the placeholder)
        if t.get("value") is None:
            t["value_eur"] = None
    for a in awards:
        if a.get("value") is None:
            a["value_eur"] = None
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


if __name__ == "__main__":
    print("placeholder rows set to NULL:", migrate_placeholders())
