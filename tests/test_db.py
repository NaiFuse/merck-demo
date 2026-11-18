"""One test per table: insert a row, check the count, check the key constraint."""

import duckdb
import pytest

from src.db import TABLES, init_db, table_counts


def test_documents(con):
    con.execute(
        "INSERT INTO documents (source, url_or_path, retrieved_at) VALUES (?, ?, now())",
        ["manual_pdf", "data/raw/merck/2026-Q3-Press-Release-EN.pdf"],
    )
    row = con.execute("SELECT id, source FROM documents").fetchone()
    assert row == (1, "manual_pdf")  # id comes from the sequence
    assert table_counts(con)["documents"] == 1


def test_facts(con):
    doc_id = con.execute(
        "INSERT INTO documents (source, url_or_path, retrieved_at) "
        "VALUES ('edgar', 'https://example.org/10-k', now()) RETURNING id"
    ).fetchone()[0]
    con.execute(
        "INSERT INTO facts (document_id, entity, period, metric, value, unit, region, segment, quote, grade) "
        "VALUES (?, 'Avantor', 'FY2025', 'revenue', 6800, 'USD_m', 'EMEA', NULL, 'Revenue was $6.8 billion', 'A')",
        [doc_id],
    )
    assert table_counts(con)["facts"] == 1
    with pytest.raises(duckdb.ConstraintException):  # grade must be A/B/C
        con.execute(
            "INSERT INTO facts (document_id, entity, period, metric, value, unit, quote, grade) "
            "VALUES (?, 'x', 'FY2025', 'm', 1, 'u', 'q', 'D')",
            [doc_id],
        )
    with pytest.raises(duckdb.ConstraintException):  # document must exist
        con.execute(
            "INSERT INTO facts (document_id, entity, period, metric, value, unit, quote, grade) "
            "VALUES (999, 'x', 'FY2025', 'm', 1, 'u', 'q', 'A')"
        )


def test_assumptions(con):
    con.execute(
        "INSERT INTO assumptions (key, value_low, value_base, value_high, source, grade, note) "
        "VALUES ('academia_consumables_intensity', 0.03, 0.05, 0.07, 'placeholder', 'C', 'test row')"
    )
    assert table_counts(con)["assumptions"] == 1
    with pytest.raises(duckdb.ConstraintException):  # key is unique
        con.execute(
            "INSERT INTO assumptions (key, value_base, source, grade) "
            "VALUES ('academia_consumables_intensity', 1, 's', 'C')"
        )


def test_tenders(con):
    con.execute(
        "INSERT INTO tenders (notice_id, country, buyer, cpv, value_eur, award_date, winner_name, winner_id) "
        "VALUES ('123456-2025', 'DE', 'Universität X', '33696300', 250000.0, DATE '2025-03-01', 'Example GmbH', 'DE123')"
    )
    assert table_counts(con)["tenders"] == 1
    with pytest.raises(duckdb.ConstraintException):  # notice_id is the primary key
        con.execute("INSERT INTO tenders (notice_id) VALUES ('123456-2025')")


def test_market_cells(con):
    con.execute(
        "INSERT INTO market_cells VALUES ('academia', 'lab_consumables', 'dach', 2025, 'A', 90, 100, 110, 'C')"
    )
    assert table_counts(con)["market_cells"] == 1
    with pytest.raises(duckdb.ConstraintException):  # one row per segment/category/cluster/year/method
        con.execute(
            "INSERT INTO market_cells VALUES ('academia', 'lab_consumables', 'dach', 2025, 'A', 1, 2, 3, 'C')"
        )


def test_schema_is_idempotent_and_complete(con):
    init_db(con)  # second run must not fail
    assert table_counts(con) == {t: 0 for t in TABLES}
