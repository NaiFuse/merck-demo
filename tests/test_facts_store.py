"""facts_store: batched, idempotent writes; rule-1 gate; never touches tender tables (tmp DB only)."""

from datetime import datetime

import pytest

from src.db import connect
from src.facts_store import AssumptionRow, FactRow, check_rows, save

TS = datetime(2026, 11, 18)


def row(value=11826.0, url="https://sec/x_htm.xml", quote="... | value=11826000000 USD | ...", **kw):
    base = dict(source="edgar", url_or_path=url, retrieved_at=TS, entity="thermo_fisher", period="FY2025",
                metric="revenue", value=value, unit="USD_m", region="Europe", segment=None, quote=quote, grade="A")
    return FactRow(**(base | kw))


def pdf_row():
    return FactRow("manual_pdf", "data/raw/peers/x.pdf", TS, "thermo_fisher", "2026Q3", "organic_growth", 5.0, "%",
                   None, None, "Revenue grew 5% to $11.12 billion", "A", {"page": 2})


def test_save_is_idempotent_and_leaves_tenders_alone(tmp_path):
    db = tmp_path / "t.duckdb"
    with connect(db) as con:
        from src.db import init_db
        init_db(con)
        con.execute("INSERT INTO tenders (notice_id, source) VALUES ('FTS-1', 'uk_fts')")
    facts = [row(), pdf_row()]
    assumptions = [AssumptionRow("thermo_fisher.ds_share", 0.45, 0.55, 0.65, "estimate", "C")]
    save(facts, assumptions, db)
    out = save(facts, assumptions, db)
    c = out["table_counts"]
    assert (c["documents"], c["facts"], c["assumptions"], c["tenders"]) == (2, 2, 1, 1)
    with connect(db) as con:
        assert con.execute("SELECT notice_id FROM tenders").fetchall() == [("FTS-1",)]
        assert con.execute("SELECT quote FROM facts WHERE unit = '%'").fetchone()[0].startswith("Revenue grew 5%")


def test_reload_replaces_a_documents_facts(tmp_path):
    db = tmp_path / "t.duckdb"
    save([row(), row(value=8781.0, region=None, segment="segment=LSS", quote="value=8781000000 USD")], db_path=db)
    out = save([row()], db_path=db)
    assert out["table_counts"]["facts"] == 1


def test_rule_1_gate_blocks_bad_quote(tmp_path):
    with pytest.raises(ValueError, match="quote does not contain"):
        save([row(value=12000.0)], db_path=tmp_path / "t.duckdb")
    assert not (tmp_path / "t.duckdb").exists()  # gate runs before the DB is even opened


def test_check_rows_rejects_bad_grade():
    with pytest.raises(ValueError, match="grade"):
        check_rows([row(grade="D")])


def test_failed_batch_rolls_back(tmp_path):
    db = tmp_path / "t.duckdb"
    save([row()], db_path=db)
    bad = [row(url="https://sec/y_htm.xml"), row(url="https://sec/y_htm.xml", entity=None)]  # NOT NULL violation
    with pytest.raises(Exception):
        save(bad, db_path=db)
    with connect(db) as con:
        assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 1



def test_retired_assumptions_are_removed(tmp_path):
    db = tmp_path / "t.duckdb"
    save([row()], [AssumptionRow("thermo_fisher.academia_share", 0.18, 0.24, 0.30, "old", "C"),
                   AssumptionRow("market.x", 1, 1, 1, "m", "C")], db)
    out = save([row()], [AssumptionRow("thermo_fisher.ag_share_company", 0.14, 0.14, 0.14, "ARS", "A")], db)
    with connect(db) as con:
        keys = {k for (k,) in con.execute("SELECT key FROM assumptions").fetchall()}
    assert keys == {"thermo_fisher.ag_share_company", "market.x"}   # other owners untouched
