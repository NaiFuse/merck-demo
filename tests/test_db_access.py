"""Dashboard access is read-only; write access stays with loader code."""

import duckdb
import pytest

from src.db import connect, connect_readonly, init_db, table_counts


def test_readonly_connection_cannot_write(tmp_path):
    path = tmp_path / "ro.duckdb"
    with connect(path) as con:
        init_db(con)
    with connect_readonly(path) as ro:
        assert table_counts(ro)["documents"] == 0
        with pytest.raises(duckdb.InvalidInputException):
            ro.execute("INSERT INTO tenders (notice_id) VALUES ('x')")


def test_readonly_on_missing_db_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.db"):
        connect_readonly(tmp_path / "missing.duckdb")
    assert not (tmp_path / "missing.duckdb").exists()  # must not create an empty DB


def test_app_does_not_open_writable_connection():
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "src" / "app"
    files = sorted(app.rglob("*.py"))
    assert any("connect_readonly" in f.read_text() for f in files)
    for f in files:
        src = f.read_text()
        assert "init_db" not in src and "connect(" not in src.replace("connect_readonly(", ""), f.name
