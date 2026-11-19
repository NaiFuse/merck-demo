"""DuckDB connection and schema. Schema creation is idempotent."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from src.settings import get_settings

TABLES = ["documents", "facts", "assumptions", "tenders", "tender_awards", "fx_rates", "market_cells"]

SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS documents_id_seq;
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY DEFAULT nextval('documents_id_seq'),
    source        VARCHAR NOT NULL,           -- e.g. 'edgar', 'ted', 'manual_pdf'
    url_or_path   VARCHAR NOT NULL,           -- source URL, or data/raw path (+ page in facts.quote context)
    retrieved_at  TIMESTAMP NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS facts_id_seq;
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY DEFAULT nextval('facts_id_seq'),
    document_id  INTEGER NOT NULL REFERENCES documents(id),
    entity       VARCHAR NOT NULL,
    period       VARCHAR NOT NULL,            -- e.g. 'FY2025', '2026Q3'
    metric       VARCHAR NOT NULL,
    value        DOUBLE NOT NULL,
    unit         VARCHAR NOT NULL,            -- e.g. 'EUR_m', 'USD_m', '%'
    region       VARCHAR,
    segment      VARCHAR,
    quote        VARCHAR NOT NULL,            -- exact quote/row the value came from
    grade        VARCHAR NOT NULL CHECK (grade IN ('A', 'B', 'C'))
);

CREATE SEQUENCE IF NOT EXISTS assumptions_id_seq;
CREATE TABLE IF NOT EXISTS assumptions (
    id          INTEGER PRIMARY KEY DEFAULT nextval('assumptions_id_seq'),
    key         VARCHAR NOT NULL UNIQUE,
    value_low   DOUBLE,
    value_base  DOUBLE NOT NULL,
    value_high  DOUBLE,
    source      VARCHAR NOT NULL,
    grade       VARCHAR NOT NULL CHECK (grade IN ('A', 'B', 'C')),
    note        VARCHAR
);

CREATE TABLE IF NOT EXISTS tenders (
    notice_id    VARCHAR PRIMARY KEY,         -- TED publication number, e.g. '8729-2025'
    country      VARCHAR,                     -- buyer country, ISO-2
    buyer        VARCHAR,
    cpv          VARCHAR,                     -- matched CPV codes from config/ted_cpv.yaml, ';'-joined
    value_eur    DOUBLE,                      -- notice award value converted to EUR (ECB monthly rate)
    award_date   DATE,
    winner_name  VARCHAR,                     -- all winners, ';'-joined (per-winner rows: tender_awards)
    winner_id    VARCHAR
);

CREATE TABLE IF NOT EXISTS tender_awards (
    notice_id    VARCHAR NOT NULL,
    seq          INTEGER NOT NULL,            -- position of the winner within the notice
    winner_name  VARCHAR,
    winner_id    VARCHAR,
    value        DOUBLE,                      -- in original currency
    currency     VARCHAR,
    value_eur    DOUBLE,
    alignment    VARCHAR NOT NULL,            -- 'single' | 'lot' | 'split_equal' | 'no_value' | 'xml'
    grade        VARCHAR NOT NULL CHECK (grade IN ('A', 'B', 'C')),
    PRIMARY KEY (notice_id, seq)
);

CREATE TABLE IF NOT EXISTS fx_rates (
    currency       VARCHAR NOT NULL,
    month          VARCHAR NOT NULL,          -- 'YYYY-MM'
    units_per_eur  DOUBLE NOT NULL,           -- ECB reference rate, monthly average
    source_url     VARCHAR NOT NULL,
    retrieved_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (currency, month)
);

CREATE TABLE IF NOT EXISTS market_cells (
    segment     VARCHAR NOT NULL,
    category    VARCHAR NOT NULL,
    cluster     VARCHAR NOT NULL,
    year        INTEGER NOT NULL,
    method      VARCHAR NOT NULL,             -- 'A' supply side, 'B' demand side, 'reconciled' (A vs B, 15% rule)
    value_low   DOUBLE,
    value_base  DOUBLE,
    value_high  DOUBLE,
    grade       VARCHAR NOT NULL CHECK (grade IN ('A', 'B', 'C')),
    PRIMARY KEY (segment, category, cluster, year, method)
);
"""

# Extra tenders columns (provenance + raw values). ALTER ... IF NOT EXISTS migrates existing DBs.
TENDERS_EXTRA_COLUMNS = {
    "notice_type": "VARCHAR",
    "publication_date": "DATE",
    "buyer_legal_type": "VARCHAR",
    "cpv_all": "VARCHAR",          # all CPV codes on the notice
    "value": "DOUBLE",             # award value in original currency
    "currency": "VARCHAR",
    "value_field": "VARCHAR",      # TED field the value was read from
    "n_winners": "INTEGER",
    "source_url": "VARCHAR",       # human-readable TED notice page
    "api_query": "VARCHAR",        # exact TED expert query that returned the notice
    "retrieved_at": "TIMESTAMP",
    "raw_json": "VARCHAR",         # exact API row (provenance)
    "is_framework": "BOOLEAN",     # framework agreement: values are ceilings, not spend
    "title": "VARCHAR",            # procedure title
    "source": "VARCHAR",           # 'ted' | 'uk_fts'
}
MARKET_CELLS_EXTRA_COLUMNS = {
    "status": "VARCHAR",           # 'estimate' (A/B) | 'check' (anchor/floor) | 'published' (A, B within 15%) |
                                   # 'range' (overlap of A and B) | 'range_no_overlap' (min P10 .. max P90)
    "gap_pct": "DOUBLE",           # |A50 - B50| / mean(A50, B50) on reconciled rows
    "note": "VARCHAR",
    "computed_at": "TIMESTAMP",
}
TENDER_AWARDS_EXTRA_COLUMNS = {
    "evidence": "VARCHAR",         # exact field/element the value came from
    "source_url": "VARCHAR",       # notice XML URL when the row came from XML, else NULL (see tenders.source_url)
    "scope_note": "VARCHAR",       # 'lab_water' | 'lab_water_possible' | NULL
}


def _db_path(path: Path | str | None) -> Path:
    return Path(path) if path else get_settings().duckdb_path


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Writable connection – for loader/ETL code only. Keep it short-lived (open, write, close):
    DuckDB allows one writing process, and any open connection elsewhere blocks it."""
    db_path = _db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def connect_readonly(
    path: Path | str | None = None, retries: int = 5, wait_s: float = 0.4
) -> duckdb.DuckDBPyConnection:
    """Read-only connection for the dashboard. Retries briefly while a loader holds the write lock.

    Raises FileNotFoundError if the DB was never created, duckdb.IOException/ConnectionException if still locked.
    """
    db_path = _db_path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist – run: uv run python -m src.db")
    for attempt in range(retries):
        try:
            return duckdb.connect(str(db_path), read_only=True)
        except (duckdb.IOException, duckdb.ConnectionException):
            if attempt == retries - 1:
                raise
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)
    for col, typ in TENDERS_EXTRA_COLUMNS.items():
        con.execute(f"ALTER TABLE tenders ADD COLUMN IF NOT EXISTS {col} {typ}")
    con.execute("UPDATE tenders SET source = 'ted' WHERE source IS NULL")  # rows loaded before 'source' existed
    for col, typ in TENDER_AWARDS_EXTRA_COLUMNS.items():
        con.execute(f"ALTER TABLE tender_awards ADD COLUMN IF NOT EXISTS {col} {typ}")
    for col, typ in MARKET_CELLS_EXTRA_COLUMNS.items():
        con.execute(f"ALTER TABLE market_cells ADD COLUMN IF NOT EXISTS {col} {typ}")


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


if __name__ == "__main__":
    with connect() as con:
        init_db(con)
        for table, n in table_counts(con).items():
            print(f"{table:<14}{n:>6}")
