"""DuckDB connection and schema. Schema creation is idempotent."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from src.settings import get_settings

TABLES = ["documents", "facts", "assumptions", "tenders", "market_cells"]

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
    notice_id    VARCHAR PRIMARY KEY,
    country      VARCHAR,
    buyer        VARCHAR,
    cpv          VARCHAR,
    value_eur    DOUBLE,
    award_date   DATE,
    winner_name  VARCHAR,
    winner_id    VARCHAR
);

CREATE TABLE IF NOT EXISTS market_cells (
    segment     VARCHAR NOT NULL,
    category    VARCHAR NOT NULL,
    cluster     VARCHAR NOT NULL,
    year        INTEGER NOT NULL,
    method      VARCHAR NOT NULL,             -- 'A' supply side, 'B' demand side, 'published'
    value_low   DOUBLE,
    value_base  DOUBLE,
    value_high  DOUBLE,
    grade       VARCHAR NOT NULL CHECK (grade IN ('A', 'B', 'C')),
    PRIMARY KEY (segment, category, cluster, year, method)
);
"""


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


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


if __name__ == "__main__":
    with connect() as con:
        init_db(con)
        for table, n in table_counts(con).items():
            print(f"{table:<14}{n:>6}")
