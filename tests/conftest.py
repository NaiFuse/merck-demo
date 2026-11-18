import pytest

from src.db import connect, init_db


@pytest.fixture
def con(tmp_path):
    """Fresh DuckDB with the full schema, in a temp dir (never touches data/demo.duckdb)."""
    c = connect(tmp_path / "test.duckdb")
    init_db(c)
    yield c
    c.close()
