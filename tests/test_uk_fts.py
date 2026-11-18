"""UK FTS connector: CPV filter, dedupe, award grading, provenance, paging, Retry-After, rule 7 (no network)."""

import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from src.connectors import tender_store, uk_fts
from src.db import connect

PKG = json.loads((Path(__file__).parent / "fixtures" / "fts_release_min.json").read_text(encoding="utf-8"))
REL = {r["id"]: r for r in PKG["releases"]}
CPV = {"33696500", "38437110"}
TS = datetime(2026, 11, 18, 12, 0)


def test_cpv_filter_uses_main_and_additional_classifications():
    assert uk_fts.release_cpvs(REL["010000-2025"]) == ["33696500", "38437110"]
    assert uk_fts.keep_release(REL["010000-2025"], {"38437110"}, date(2023, 11, 18), date(2026, 11, 18))
    assert not uk_fts.keep_release(REL["030000-2025"], CPV, date(2023, 11, 18), date(2026, 11, 18))
    assert not uk_fts.keep_release(REL["010000-2025"], CPV, date(2025, 6, 1), date(2026, 11, 18))  # outside window


def test_award_values_come_from_contracts_by_award_id():
    v = uk_fts.award_values(REL["010000-2025"])
    assert v["010000-2025-1"][:2] == (500000.0, "GBP")
    assert "contracts[0].value.amount=500000" in v["010000-2025-1"][2]


def test_dedupe_keeps_latest_release_per_lot():
    out = {r["id"]: kept for r, kept in uk_fts.dedupe(PKG["releases"])}
    assert out["020000-2025"] == {"020000-2025-1"}                       # newer contract release wins lot 1
    assert out["010000-2025"] == {"010000-2025-2", "010000-2025-3"}      # lots 2 and 3 only from the older one
    assert out["030000-2025"] == {"030000-2025-1"}


def test_parse_release_grades_and_provenance():
    t, a = uk_fts.parse_release(REL["010000-2025"], {"010000-2025-1", "010000-2025-2", "010000-2025-3"}, "U", TS, CPV)
    assert t["notice_id"] == "FTS-010000-2025" and t["source"] == "uk_fts" and t["country"] == "GB"
    assert t["source_url"] == "https://www.find-tender.service.gov.uk/Notice/010000-2025"
    assert t["value"] == 730000 and t["currency"] == "GBP" and t["is_framework"] is True
    assert t["award_date"] == date(2025, 2, 1) and t["publication_date"] == date(2025, 3, 1)
    assert t["cpv"] == "33696500;38437110" and t["api_query"] == "U" and "Milli-Q" in t["raw_json"]
    got = [(x["winner_name"], x["value"], x["alignment"], x["grade"]) for x in a]
    assert got == [("Fisher Scientific UK Ltd", 500000.0, "lot", "A"),
                   ("VWR International Ltd", 100000.0, "split_equal", "C"),
                   ("Starlab (UK) Ltd", 100000.0, "split_equal", "C"),
                   ("Merck Life Science UK Limited", 30000.0, "lot", "A")]
    assert a[0]["winner_id"] == "GB-COH:00011111"


def test_lab_water_tagged_on_merck_lot():
    t, a = uk_fts.parse_release(REL["010000-2025"], {"010000-2025-3"}, "U", TS, CPV)
    tender_store.apply_scope_flags([t], a)
    assert a[0]["scope_note"] == "lab_water" and t["value"] == 30000


def test_verify_total_rejects_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        uk_fts.verify_total(REL["010000-2025"], {"010000-2025-1"}, 1.0)


def test_month_windows_clip_to_window():
    w = uk_fts.month_windows(date(2023, 11, 18), date(2024, 1, 5))
    assert [(l, a.date(), b.date()) for l, a, b in w] == [
        ("2023-11", date(2023, 11, 18), date(2023, 12, 1)),
        ("2023-12", date(2023, 12, 1), date(2024, 1, 1)),
        ("2024-01", date(2024, 1, 1), date(2024, 1, 6))]


def _client(responses):
    calls, slept = [], []

    def handler(request):
        calls.append(str(request.url))
        status, headers, body = responses.pop(0)
        return httpx.Response(status, headers=headers, json=body)

    c = uk_fts.FtsClient(httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0, sleep=slept.append)
    return c, calls, slept


def test_paging_follows_links_next_and_honours_retry_after():
    page2 = {"releases": [REL["030000-2025"]]}
    c, calls, slept = _client([(200, {}, PKG), (429, {"Retry-After": "7"}, {}), (200, {}, page2)])
    pages = list(uk_fts.iter_pages(c, datetime(2025, 3, 1), datetime(2025, 4, 1)))
    assert len(pages) == 2 and "cursor=abc" in calls[1] and 7.0 in slept
    assert "stages=award" in calls[0] and "updatedFrom=2025-03-01T00%3A00%3A00" in calls[0]


def test_run_fetch_never_holds_db_during_network_and_writes_deduped_rows(tmp_path, monkeypatch):
    state = {"open": 0, "network_while_open": 0}

    class TrackedCon:
        def __init__(self, con):
            self.con = con

        def __enter__(self):
            state["open"] += 1
            return self.con.__enter__()

        def __exit__(self, *exc):
            state["open"] -= 1
            return self.con.__exit__(*exc)

    real_connect = tender_store.connect
    monkeypatch.setattr(tender_store, "connect", lambda p=None: TrackedCon(real_connect(p)))

    class FakeClient:
        def get(self, url, params=None):
            state["network_while_open"] += state["open"] > 0
            if params and params["updatedFrom"].startswith("2025-05"):
                return url, {"releases": PKG["releases"]}
            return url, {"releases": []}

        def close(self):
            pass

    def fake_fx(currencies, start, client=None):
        state["network_while_open"] += state["open"] > 0
        return [uk_fts.ecb_fx.FxRate("GBP", m, 0.85, "ecb", TS) for m in ("2025-02", "2025-03", "2025-05")]

    monkeypatch.setattr(uk_fts.ecb_fx, "fetch_monthly_rates", fake_fx)
    db = tmp_path / "f.duckdb"
    out = uk_fts.run_fetch(today=date(2026, 11, 18), db_path=db, cache_dir=tmp_path / "cache", client=FakeClient())
    assert state["network_while_open"] == 0
    assert out["notices"] == 2 and out["releases_matched"] == 2
    with connect(db) as con:
        assert con.execute("SELECT count(*) FROM tenders WHERE source = 'uk_fts'").fetchone()[0] == 2
        v = con.execute("SELECT round(sum(value_eur)) FROM tender_awards").fetchone()[0]
        assert v == round((480000 + 200000 + 30000) / 0.85)  # lot 1 counted once (latest release)
    assert (tmp_path / "cache" / "2025-05.jsonl").exists()
    # second run reuses the cache for completed months
    out2 = uk_fts.run_fetch(today=date(2026, 11, 18), db_path=db, cache_dir=tmp_path / "cache", client=FakeClient())
    assert out2["scanned_this_run"] == 0 and out2["notices"] == 2


def test_cache_only_makes_no_fts_calls_and_reports_coverage(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    cpv = set(uk_fts.load_cpv_config()["include"])
    rows = [{"api_url": "u", "retrieved_at": TS.isoformat(), "release": r} for r in PKG["releases"]
            if uk_fts.keep_release(r, cpv, date(2023, 11, 18), date(2026, 11, 18))]  # what fetch_month caches
    (cache / "2025-05.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (cache / "2025-04.jsonl").write_text("", encoding="utf-8")       # fully paged, 0 matches
    (cache / "_manifest.json").write_text(json.dumps({"2025-04": {"scanned": 100}, "2025-05": {"scanned": 1000}}))
    monkeypatch.setattr(uk_fts.ecb_fx, "fetch_monthly_rates",
                        lambda c, s, client=None: [uk_fts.ecb_fx.FxRate("GBP", m, 0.85, "ecb", TS)
                                                   for m in ("2025-02", "2025-03", "2025-05")])
    monkeypatch.setattr(uk_fts, "FtsClient", lambda *a, **k: pytest.fail("FTS client must not be created"))
    out = uk_fts.run_fetch(today=date(2026, 11, 18), db_path=tmp_path / "f.duckdb", cache_dir=cache, cache_only=True)
    assert out["notices"] == 2 and out["scanned_this_run"] == 0
    cov = out["coverage"]
    assert (cov["first_month"], cov["last_month"], cov["end"]) == ("2025-04", "2025-05", date(2025, 5, 31))
    assert cov["missing_months"] == [] and cov["low_scan_months"] == ["2025-04"]


def test_coverage_reports_gaps(tmp_path):
    for m in ("2025-01", "2025-03"):
        (tmp_path / f"{m}.jsonl").write_text("")
    cov = uk_fts.coverage(tmp_path)
    assert cov["missing_months"] == ["2025-02"] and cov["low_scan_months"] == []
    (tmp_path / "none").mkdir()
    assert uk_fts.coverage(tmp_path / "none") is None



def test_placeholder_values_stored_as_null(tmp_path):
    t = {"notice_id": "x", "value": -1.0, "value_field": "total-value"}
    a = {"notice_id": "x", "value": -1.0, "alignment": "single", "grade": "A", "evidence": "total-value=-1"}
    ok = {"notice_id": "y", "value": 5.0, "alignment": "single", "grade": "A", "evidence": "e"}
    assert tender_store.null_placeholder_values([t], [a, ok]) == 2
    assert t["value"] is None and "raw=-1.0" in t["value_field"]
    assert (a["value"], a["alignment"], a["grade"]) == (None, "no_value", "C") and "placeholder" in a["evidence"]
    assert ok["value"] == 5.0


def test_migrate_placeholders(tmp_path):
    db = tmp_path / "m.duckdb"
    with connect(db) as con:
        from src.db import init_db
        init_db(con)
        con.execute("INSERT INTO tenders (notice_id, value, value_eur) VALUES ('a', -1, -1), ('b', 10, 10)")
        con.execute("INSERT INTO tender_awards (notice_id, seq, value, value_eur, alignment, grade) "
                    "VALUES ('a', 0, -1, -1, 'single', 'A'), ('b', 0, 10, 10, 'single', 'A')")
    assert tender_store.migrate_placeholders(db) == {"tenders": 1, "awards": 1}
    with connect(db) as con:
        assert con.execute("SELECT value, alignment, grade FROM tender_awards WHERE notice_id='a'").fetchone() == (None, "no_value", "C")
        assert con.execute("SELECT value FROM tender_awards WHERE notice_id='b'").fetchone() == (10,)
