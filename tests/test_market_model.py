"""Research base panel/shares and market model (method A, B, reconciliation, market_cells writes) – offline."""

from datetime import datetime

import numpy as np
import pytest

from src.db import connect
from src.facts_store import FactRow, check_cells, save
from src.models import market_model as mm
from src.models import peer_restatement as pr
from src.models import research_base as rb

TS = datetime(2026, 11, 18)


def f(source, c, year, metric, value, unit, segment):
    return FactRow(source, "u", TS, c, str(year), metric, value, unit, c, segment, "q", "A")


def fixed(key, v):
    return {"key": key, "low": v, "base": v, "high": v, "grade": "A", "source": "test"}


# ---------------------------------------------------------------- research base

def test_fill_interior_and_window_mean():
    assert rb.fill_interior({2019: 10.0, 2022: 40.0}) == {2019: 10.0, 2020: 20.0, 2021: 30.0, 2022: 40.0}
    assert rb.window_mean({2019: 10.0, 2021: 30.0}, 2019, 2021) == (20.0, False)
    assert rb.window_mean({2019: 10.0}, 2019, 2021) == (10.0, True)          # carried forward
    assert rb.window_mean({}, 2019, 2021) == (0.0, False)


def test_panel_prefers_oecd_for_uk_and_converts_gbp(monkeypatch):
    monkeypatch.setattr(rb, "_gbp_rates", lambda years, online: {y: 0.8 for y in years})
    facts = [f("eurostat", "GB", 2019, "rd_expenditure", 9000.0, "EUR_m", "HES"),      # stale -> dropped
             f("eurostat", "GB", 2019, "rd_expenditure", 1000.0, "EUR_m", "PNP"),      # no OECD series -> kept
             f("oecd_msti", "GB", 2023, "rd_expenditure", 16000.0, "GBP_m", "HES"),
             f("eurostat", "DE", 2023, "rd_expenditure", 23000.0, "EUR_m", "HES"),
             f("openalex", "DE", 2023, "publications", 5.0, "count", "a"),
             f("openalex", "DE", 2023, "publications", 7.0, "count", "b"),
             f("cordis", "DE", 2019, "eu_grant_net_contribution", 3.0, "EUR", "HES"),
             f("edgar", "x", 2023, "rd_expenditure", 1.0, "EUR_m", "HES")]               # other source ignored
    p = rb.build_panel(facts)
    assert p.spend[("GB", "HES")] == {2023: pytest.approx(20000.0)} and p.spend_source[("GB", "HES")] == "oecd_msti"
    assert p.spend[("GB", "PNP")] == {2019: 1000.0}
    assert p.publications["DE"] == {2023: 12.0} and p.grants["DE"] == {2019: 3.0}


def _panel():
    p = rb.Panel()
    p.spend = {("DE", "HES"): {2021: 60.0, 2022: 60.0, 2023: 60.0}, ("FR", "HES"): {2021: 30.0, 2023: 30.0},
               ("IT", "HES"): {2021: 10.0, 2022: 10.0, 2023: 10.0}}
    p.researchers = {("DE", "HES"): {2021: 50.0}, ("FR", "HES"): {2021: 40.0}, ("IT", "HES"): {2021: 10.0}}
    p.publications = {"DE": {2023: 40.0}, "FR": {2023: 40.0}, "IT": {2023: 20.0}}
    p.grants = {"DE": {2019: 70.0}, "FR": {2019: 20.0}, "IT": {2019: 10.0}}
    return p


IND = {"rd_spend": {"window": [2021, 2023]}, "researchers": {"window": [2021, 2023]},
       "publications": {"window": [2023, 2023]}, "eu_grants": {"window": [2019, 2019]}}


def test_group_shares_and_assumption_range():
    ind = rb.group_shares(_panel(), IND, {"DE": ["DE"]}, ["DE", "FR", "IT"])
    shares = {s.indicator: s.shares["DE"] for s in ind}
    assert shares == pytest.approx({"rd_spend": 0.6, "researchers": 0.5, "publications": 0.4, "eu_grants": 0.7})
    a = rb._assumption("share_de", "DE", ind, "rd_spend", "test")
    assert (a.low, a.base, a.high, a.grade) == (0.4, 0.6, 0.7, "B")
    assert "researchers 50.0%" in a.source


def test_apply_to_market_fills_clusters_and_germany_share(monkeypatch):
    monkeypatch.setattr(rb, "build_panel", lambda facts, online_fx=False: _panel())
    monkeypatch.setattr(rb, "load_taxonomy", lambda: {"clusters": {"dach": {"countries": ["DE"]},
                                                                    "fr": {"countries": ["FR"]}}})
    market = pr.Market.model_validate({"fx": {}, "cluster_shares_from": "research_base",
                                       "research_base": {"base_indicator": "rd_spend", "indicators": IND},
                                       "dach_anchor": {"x": fixed("x", 0.5)}})
    monkeypatch.setattr("src.connectors.eurostat.load_config", lambda: {"europe_countries": ["DE", "FR", "IT"]})
    m = rb.apply_to_market(market, [])
    assert set(m.clusters) == {"dach", "fr"} and m.clusters["dach"].base == pytest.approx(0.6)
    assert m.dach_anchor["germany_share_of_dach"].base == 1.0          # DE is all of 'DACH' here


# ---------------------------------------------------------------- method B

def test_demand_terms_and_growth():
    p = rb.Panel(spend={("DE", "HES"): {2019: 100.0, 2023: 146.41}, ("DE", "PNP"): {2022: 10.0},
                        ("FR", "HES"): {2019: 50.0}})
    t = {(x.sector, x.years_ahead): x.value for x in mm.demand_terms(p, ["DE"], 2025)}
    assert t == {("HES", 2): 146.41, ("PNP", 3): 10.0}
    assert mm.demand_terms(p, ["DE"], 2020)[0].value == pytest.approx(111.6025)  # interpolated; PNP not yet started
    assert mm.cluster_growth(p, ["DE"], (2019, 2023)) == pytest.approx(0.10)   # PNP not observed at both ends


def _market(intensity=0.02, others=0.5):
    return pr.Market.model_validate({
        "fx": {}, "monte_carlo": {"draws": 200, "seed": 1},
        "clusters": {"dach": fixed("c", 1.0)},
        "demand": {"years": [2023, 2025], "growth_window": [2022, 2023],
                   "growth_offset": fixed("g", 0.0), "intensity": fixed("i", intensity),
                   "non_rd_uplift": fixed("u", 0.1),
                   "sector_weights": {"HES": fixed("wh", 1.0), "GOV": fixed("wg", 0.5), "PNP": fixed("wp", 0.5)}},
        "supply": {"year": 2025, "others_share": fixed("o", others)}})


def _taxonomy(monkeypatch):
    monkeypatch.setattr(mm, "load_taxonomy", lambda: {"clusters": {"dach": {"countries": ["DE"]}},
                                                       "publish_tolerance": 0.15})


def test_method_b_hand_computed(monkeypatch):
    _taxonomy(monkeypatch)
    p = rb.Panel(spend={("DE", "HES"): {2022: 1000.0, 2023: 1100.0}, ("DE", "GOV"): {2022: 400.0, 2023: 440.0}})
    cells = {(c.cluster, c.year): c for c in mm.method_b(p, _market(), 200, 1)}
    c23, c25 = cells[("dach", 2023)], cells[("dach", 2025)]
    assert c23.pct[1] == pytest.approx(0.02 * (1100 + 0.5 * 440) * 1.1)
    assert c25.pct[1] == pytest.approx(0.02 * (1100 + 0.5 * 440) * 1.1 * 1.1 ** 2)    # nowcast at 10% CAGR
    assert c25.nowcast_share == 1.0 and c23.nowcast_share == 0.0
    assert cells[("total", 2025)].pct[1] == pytest.approx(c25.pct[1])


# ---------------------------------------------------------------- method A + reconciliation

def _peers(market, dach_values):
    res = []
    for i, v in enumerate(dach_values):
        r = pr.Restatement(f"p{i}", f"P{i}", "FY2025", [], (0, 0, 0), {"dach": (v, v, v)}, {}, 1.0, "", "A")
        r.mc_draws = {"dach": np.full(200, v), "total": np.full(200, v)}
        r.mc = {"dach": (v, v, v), "total": (v, v, v)}
        res.append(r)
    return pr.RunResult(res, [], market)


def test_method_a_scales_peers_by_others_share():
    m = _market(others=0.6)
    cells = {c.cluster: c for c in mm.method_a(_peers(m, [30.0, 10.0]), m, 200, 1)}
    assert cells["dach"].pct[1] == pytest.approx(40.0 / 0.4)
    assert cells["dach"].peers == {"p0": (30.0, 30.0, 30.0), "p1": (10.0, 10.0, 10.0)}


def test_reconcile_publishes_within_tolerance_else_range():
    a = mm.SupplyCell("dach", 2025, {}, np.linspace(90, 110, 101))
    near = mm.DemandCell("dach", 2025, 0, 0, 0, 0, np.linspace(100, 120, 101))
    far = mm.DemandCell("dach", 2025, 0, 0, 0, 0, np.linspace(200, 300, 101))
    ok = mm.reconcile(a, near, 0.15)
    assert (ok.status, ok.grade) == ("published", "B") and ok.base == pytest.approx(105.0, abs=0.5)
    assert ok.gap_pct == pytest.approx(10 / 105)
    bad = mm.reconcile(a, far, 0.15)
    assert (bad.status, bad.grade, bad.base) == ("range_no_overlap", "C", None)   # A, B do not overlap: min..max
    assert bad.low == pytest.approx(92.0) and bad.high == pytest.approx(290.0)


def test_run_writes_a_b_and_reconciled_rows(monkeypatch, tmp_path):
    _taxonomy(monkeypatch)
    m = _market(intensity=0.02, others=0.5)
    p = rb.Panel(spend={("DE", "HES"): {2022: 1000.0, 2023: 1000.0}})
    # B 2025 = 0.02 x 1000 x 1.1 = 22; A = 11 / 0.5 = 22 -> published
    res = mm.run(_peers(m, [11.0]), p)
    rows = res.cell_rows()
    rec = [r for r in rows if r["method"] == "reconciled" and r["cluster"] == "dach"]
    assert len(rec) == 1 and rec[0]["status"] == "published" and rec[0]["value_base"] == pytest.approx(22.0)
    assert {(r["method"], r["year"]) for r in rows if r["cluster"] == "dach"} == {
        ("B", 2023), ("B", 2024), ("B", 2025), ("A", 2025), ("reconciled", 2025)}
    db = tmp_path / "t.duckdb"
    save([], [], db, market_cells=rows)
    out = save([], [], db, market_cells=rows)                                # idempotent
    assert out["table_counts"]["market_cells"] == len(rows)
    with connect(db) as con:
        assert con.execute("SELECT status FROM market_cells WHERE method = 'reconciled' AND cluster = 'dach'"
                           ).fetchone()[0] == "published"


def test_check_cells_enforces_ranges_for_grade_c():
    good = {"grade": "C", "value_low": 1.0, "value_base": None, "value_high": 2.0}
    check_cells([good])
    with pytest.raises(ValueError):
        check_cells([good | {"value_high": 1.0}])
