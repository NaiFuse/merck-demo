"""EDGAR connector: filing selection, XBRL parsing, period labels, provenance, cache, User-Agent, retries (offline)."""

import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from src.connectors import edgar

FIX = Path(__file__).parent / "fixtures" / "edgar"
INSTANCE = (FIX / "instance_min.xml").read_bytes()
SUBS = json.loads((FIX / "submissions_min.json").read_text())
TS = datetime(2026, 11, 18, 12, 0)


def test_select_filings_keeps_10k_10q_since_cutoff():
    fs = edgar.select_filings("thermo_fisher", SUBS)
    assert [(f.form, f.report_date) for f in fs] == [("10-Q", "2025-09-27"), ("10-K", "2025-12-31")]
    k = fs[1]
    assert k.cik == "0000097745"
    assert k.url("index.json") == "https://www.sec.gov/Archives/edgar/data/97745/000009774526000018/index.json"
    assert k.instance_guess == "tmo-20251231_htm.xml"


def test_instance_name_from_index_or_fallback():
    idx = {"directory": {"item": [{"name": "R1.htm"}, {"name": "tmo-20251231_htm.xml"}, {"name": "tmo-20251231_cal.xml"}]}}
    assert edgar.instance_name(idx, "x") == "tmo-20251231_htm.xml"
    assert edgar.instance_name({}, "fallback_htm.xml") == "fallback_htm.xml"


def test_parse_instance_reads_revenue_with_dims_and_skips_instants():
    facts = edgar.parse_instance(INSTANCE)
    assert all(f.concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" for f in facts)
    eu = next(f for f in facts if f.context_id == "c-eu")
    assert eu.dims == (("srt:StatementGeographicalAxis", "srt:EuropeMember"),)
    assert eu.value == 11826000000 and eu.unit == "USD" and eu.decimals == "-6"


@pytest.mark.parametrize("start,end,label", [
    (date(2025, 1, 1), date(2025, 12, 31), "FY2025"),
    (date(2025, 6, 29), date(2025, 9, 27), "2025Q3"),
    (date(2023, 4, 2), date(2023, 7, 1), "2023Q2"),       # 52/53-week quarter ending in early July
    (date(2025, 1, 1), date(2025, 9, 27), None),          # YTD skipped
])
def test_period_label(start, end, label):
    assert edgar.period_label(start, end) == label


def test_revenue_rows_provenance_and_filters():
    filing = edgar.select_filings("thermo_fisher", SUBS)[1]
    url = filing.url("tmo-20251231_htm.xml")
    rows = edgar.revenue_rows(filing, edgar.parse_instance(INSTANCE), url, TS)
    by = {(r.period, r.region, r.segment): r for r in rows}
    assert set(by) == {
        ("FY2025", None, None),
        ("FY2025", "Europe", None),
        ("FY2025", None, "consolidation=OperatingSegments;segment=LaboratoryProductsAndBiopharmaServices;"
                         "subsegment=ResearchAndSafetyMarketChannel"),
        ("2025Q3", None, None),
    }  # YTD and the IncomeStatementLocationAxis context are dropped
    eu = by[("FY2025", "Europe", None)]
    assert eu.value == 11826.0 and eu.unit == "USD_m" and eu.grade == "A"
    assert eu.url_or_path == url and eu.retrieved_at == TS
    assert "value=11826000000 USD" in eu.quote and "srt:EuropeMember" in eu.quote and "context=c-eu" in eu.quote


def test_latest_per_key_prefers_newest_filing():
    base = dict(source="edgar", url_or_path="u", retrieved_at=TS, entity="avantor", period="FY2024", metric="revenue",
                unit="USD_m", region=None, segment=None, quote="q", grade="A")
    old = edgar.FactRow(value=1.0, extra={"report_date": "2024-12-31"}, **base)
    new = edgar.FactRow(value=2.0, extra={"report_date": "2025-12-31"}, **base)
    assert [r.value for r in edgar.latest_per_key([new, old])] == [2.0]


def test_cross_check_against_companyfacts():
    totals = edgar.companyfacts_totals(json.loads((FIX / "companyfacts_min.json").read_text()))
    assert totals == {"FY2025": 44556.0}
    filing = edgar.select_filings("thermo_fisher", SUBS)[1]
    rows = edgar.revenue_rows(filing, edgar.parse_instance(INSTANCE), "u", TS)
    assert edgar.cross_check(rows, totals) == []
    assert edgar.cross_check(rows, {"FY2025": 40000.0})


def test_load_entity_offline_from_cache(tmp_path):
    """Pre-populated cache, client=None: no network at all."""
    subs = dict(SUBS, filings={"recent": {k: v[2:3] for k, v in SUBS["filings"]["recent"].items()}})
    filing = edgar.select_filings("thermo_fisher", subs)[0]
    files = {
        edgar.SUBMISSIONS_URL.format(cik="0000097745"): json.dumps(subs).encode(),
        filing.url("index.json"): json.dumps({"directory": {"item": [{"name": "tmo-20251231_htm.xml"}]}}).encode(),
        filing.url("tmo-20251231_htm.xml"): INSTANCE,
        filing.url("tmo-20251231.htm"): b"<html><body><p>Revenue in Europe was&nbsp;<b>$11.8 billion</b></p></body></html>",
        edgar.COMPANYFACTS_URL.format(cik="0000097745"): (FIX / "companyfacts_min.json").read_bytes(),
    }
    for url, content in files.items():
        p = edgar.cache_path(url, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    rows, problems = edgar.load_entity("thermo_fisher", None, cache_dir=tmp_path)
    assert problems == [] and len(rows) == 4
    texts = edgar.ten_k_texts("thermo_fisher", tmp_path)
    assert list(texts.values()) == ["Revenue in Europe was $11.8 billion"]


def test_offline_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        edgar.cached_get(None, "https://data.sec.gov/x.json", cache_dir=tmp_path)


def test_user_agent_required():
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        edgar.EdgarClient(user_agent="")


def test_client_sends_user_agent_and_retries_on_429():
    calls = []

    def handler(request):
        calls.append(request.headers["User-Agent"])
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    sleeps = []
    c = edgar.EdgarClient("Test Agent test@example.com", httpx.Client(transport=httpx.MockTransport(handler)),
                          min_interval=0, sleep=sleeps.append)
    assert c.get("https://data.sec.gov/x") == b"ok"
    assert calls == ["Test Agent test@example.com"] * 3
    assert sleeps[0] == 1.0  # honoured Retry-After


def test_client_does_not_retry_404():
    c = edgar.EdgarClient("T t@example.com", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
                          min_interval=0, sleep=lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("https://data.sec.gov/x")


def test_html_text_strips_tags_and_entities():
    assert edgar.html_text("<div>A&amp;B<br/>  <span>C</span></div>") == "A&B C"
