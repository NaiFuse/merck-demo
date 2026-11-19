"""Milestone 6: EUR-Lex thresholds, space-grouped numbers, plain-quote manual facts, findings arithmetic."""

import json
from datetime import datetime

import pandas as pd
import pytest

from src.facts_store import FactRow

TS = datetime(2026, 11, 19)


def f(entity, period, metric, value, unit="%", region=None, segment=None, source="manual_pdf", quote="q",
      url="u"):
    return FactRow(source, url, TS, entity, period, metric, value, unit, region, segment, quote, "A")


# ---------------------------------------------------------------- numbers

def test_space_grouped_thousands_add_a_reading():
    from src.extract.numbers import quote_contains_value as q

    assert q("in point (b), ‘EUR 143 000’ is replaced by ‘EUR 140 000’", 140000, "EUR")
    assert q("‘EUR 5 538 000’", 5538000, "EUR")
    assert q("EUR 143 000", 143, "EUR")              # the separate groups are still read
    assert not q("EUR 143 000", 144000, "EUR")


# ---------------------------------------------------------------- EUR-Lex connector

def _act_cache(tmp_path, body: str, url: str):
    from src.connectors.http_cache import cache_file

    p = cache_file(tmp_path, url, ".xhtml")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    (p.parent / (p.name + ".meta.json")).write_text(json.dumps({"url": url, "retrieved_at": TS.isoformat()}))


def test_eurlex_facts_verified_against_the_act(tmp_path):
    from src.connectors import eurlex

    url = eurlex.CELLAR.format(celex="32025R2152")
    _act_cache(tmp_path, "<html><body><p>(b) in point (b), ‘EUR 143&#160;000’ is replaced by ‘EUR 140&#160;000’;"
                         "</p></body></html>", url)
    cfg = tmp_path / "eurlex.yaml"
    cfg.write_text("acts:\n  - celex: 32025R2152\n    eli: http://data.europa.eu/eli/reg_del/2025/2152/oj\n"
                   "    title: t\n    applies: [2026, 2027]\n    facts:\n"
                   "      - {metric: procurement_threshold_supplies_central, value: 140000,\n"
                   "         quote: \"in point (b), ‘EUR 143 000’ is replaced by ‘EUR 140 000’\"}\n", encoding="utf-8")
    rows = eurlex.load_all(config=cfg, cache_dir=tmp_path)
    assert [(r.period, r.value, r.grade) for r in rows] == [("2026", 140000.0, "A"), ("2027", 140000.0, "A")]
    assert rows[0].source == "eurlex" and "CELEX 32025R2152" in rows[0].quote
    cfg.write_text(cfg.read_text().replace("value: 140000", "value: 150000"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain"):
        eurlex.load_all(config=cfg, cache_dir=tmp_path)


# ---------------------------------------------------------------- manual facts: plain quotes

def test_manual_plain_quote_verify():
    from src.extract.manual_facts import quote_text, verify

    item = {"page": 3, "value": 16.8, "unit": "%",
            "quotes": ["Herausragend war der akademische Bereich mit einem Wachstum von 16,8 %"]}
    pages = ["x", "y", "Text. Herausragend war der akademische Bereich mit einem Wachstum von 16,8 %. Mehr."]
    assert verify(item, pages) == []
    assert verify({**item, "value": 7.6}, pages)                 # wrong value rejected
    assert verify({**item, "page": 2}, pages)                    # wrong page rejected
    assert quote_text(item).startswith("p.3: Herausragend")


# ---------------------------------------------------------------- findings arithmetic

def _facts():
    rows = []
    for s, v23, v24 in (("HES", 100.0, 104.0), ("GOV", 50.0, 56.0), ("PNP", 10.0, 10.0)):
        rows += [f("DE", "2023", "rd_expenditure", v23, "EUR_m", "DE", s, "eurostat"),
                 f("DE", "2024", "rd_expenditure", v24, "EUR_m", "DE", s, "eurostat", quote="Eurostat ... (flag b)")]
    rows += [f("vwr_international_gmbh", "FY2025", "revenue", 376328, "EUR_k", "Deutschland"),
             f("vwr_international_gmbh", "FY2024", "revenue", 387182, "EUR_k", "Deutschland"),
             f("carl_roth", "FY2024", "revenue", 75984418.56, "EUR", "Inland"),
             f("carl_roth", "FY2023", "revenue", 74420811.74, "EUR", "Inland")]
    return rows


def test_derived_growth_facts():
    from src.models.findings import derived_fact_rows

    d = {(r.entity, r.segment or "", r.region or ""): r for r in derived_fact_rows(_facts())}
    assert d[("DE", "HES+GOV+PNP", "DE")].value == pytest.approx(round((170 / 160 - 1) * 100, 2))
    assert d[("DE", "HES", "DE")].value == pytest.approx(4.0)
    assert "flag" in d[("DE", "HES+GOV+PNP", "DE")].quote
    assert d[("vwr_international_gmbh", "total", "Deutschland")].value == pytest.approx(-2.8, abs=0.01)
    assert d[("carl_roth", "total", "Inland")].value == pytest.approx(2.1, abs=0.01)
    assert all(r.source == "derived" and r.grade == "A" for r in d.values())


def test_tender_shares_pooled_ranges():
    from src.models.findings import tender_shares

    rows = []
    for g, v, grade in [("thermo_fisher", 20, "A"), ("avantor", 10, "C"), ("merck", 2, "A"), ("other", 68, "C")]:
        rows.append({"group": g, "value_eur": v * 1e6, "grade": grade, "is_framework": False, "period": "2025",
                     "notice_id": g, "tier": "core", "scope_note": None, "clinical_consumables": False,
                     "academia": True, "instrument_bundle": False})
    rows.append({**rows[0], "period": "2023 partial (from 23 Nov)", "notice_id": "stub"})
    t = tender_shares(pd.DataFrame(rows))
    assert t["all"]["value"]["merck"] == pytest.approx(0.02)            # 2023 stub left out
    assert t["ab"]["value"]["merck"] == pytest.approx(2 / 22)           # A/B awards only
    assert t["merck_notices"] == 1 and t["periods"] == ["2025"]


def test_findings_numbers_trace_to_facts():
    """Every numeric figure in finding 1 comes from a stored or derived fact with a source."""
    from src.facts_store import base_facts
    from src.models import findings, merck_ds

    facts = base_facts()
    g = findings.germany_channel(facts, merck_ds.load_config())
    assert {x.grade for x in g.figures} == {"A"}
    assert all(x.sources and x.sources[0].ref for x in g.figures)
    acad = [x for x in g.figures if "Academic segment" in x.label]
    assert [round(x.value, 3) for x in acad] == [0.07, 0.076, 0.168]
    assert not any("taking" in c.lower() for c in g.caveats + [g.lead, g.title])    # no causal claim
