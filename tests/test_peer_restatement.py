"""Peer restatement: hand-computed waterfall, ranges, grades, YAML validation, evidence check, real configs (offline)."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.facts_store import FactRow
from src.models import peer_restatement as pr

TS = datetime(2026, 11, 18)


def fact(period, value, region=None, segment=None, entity="peer_x"):
    return FactRow("edgar", "u", TS, entity, period, "revenue", value, "USD_m", region, segment, "q", "A")


FACTS = [
    fact("FY2025", 1000.0),
    fact("FY2025", 300.0, region="Europe"),
    fact("FY2025", 400.0, segment="seg=Lab"),
    fact("FY2025", 200.0, segment="seg=Mixed"),
    fact("FY2025", -50.0, segment="seg=Elim"),
]


def a(key, low, base, high, grade="C", **kw):
    return {"key": key, "low": low, "base": base, "high": high, "grade": grade, "source": "test", **kw}


def cfg(**over):
    d = {
        "peer": "peer_x", "label": "Peer X", "period": "FY2025", "currency": "USD", "tenk_url": "https://sec/x.htm",
        "revenue": {"period": "FY2025"},
        "geo": {"numerator": {"period": "FY2025", "region": "Europe"}, "denominator": {"period": "FY2025"},
                "mix_factor": a("mix", 1, 1, 1, "A")},
        "scope": {
            "include": [{"ref": {"period": "FY2025", "segment": "seg=Lab"}, "weight": a("w_lab", 1, 1, 1, "A")},
                        {"ref": {"period": "FY2025", "segment": "seg=Mixed"}, "weight": a("w_mixed", 0.25, 0.5, 0.75)}],
            "deduct": [{"ref": {"period": "FY2025", "segment": "seg=Elim", "abs": True}, "weight": a("d_elim", 1, 1, 1, "A")}],
        },
        "ds_share": a("ds", 0.4, 0.5, 0.6),
        "merck_resale_share": a("merck", 0.05, 0.1, 0.15),
        "academia_share": a("acad", 0.2, 0.25, 0.3, evidence="academic and government markets"),
    }
    d.update(over)
    return pr.PeerConfig.model_validate(d)


MARKET = pr.Market.model_validate({"clusters": {"dach": a("c_dach", 0.3, 0.4, 0.5)}, "fx": {"currency": "USD"}})


def test_waterfall_hand_computed():
    r = pr.restate(cfg(), FACTS, MARKET, fx_rate=1.25)
    base = [s.value[1] for s in r.steps]
    # 1: 1000; 2: x 0.3 = 300; 3: scope (400 + 0.5*200 - 50) / 1000 = 0.45 -> 135; 4: x0.5 = 67.5; 5: x0.25 = 16.875
    assert base[:5] == pytest.approx([1000, 300, 135, 67.5, 16.875])
    # 6: x 0.4 / 1.25 EUR
    assert r.clusters["dach"][1] == pytest.approx(16.875 * 0.4 / 1.25)
    assert r.clusters_merck["dach"][1] == pytest.approx(16.875 * 0.4 / 1.25 * 0.1)


def test_ranges_are_bounding_products():
    r = pr.restate(cfg(), FACTS, MARKET, fx_rate=1.0)
    lo = 1000 * 0.3 * ((400 + 0.25 * 200 - 50) / 1000) * 0.4 * 0.2 * 0.3
    hi = 1000 * 0.3 * ((400 + 0.75 * 200 - 50) / 1000) * 0.6 * 0.3 * 0.5
    assert r.clusters["dach"][0] == pytest.approx(lo) and r.clusters["dach"][2] == pytest.approx(hi)
    assert r.total == r.clusters["dach"]


def test_deduction_uses_high_weight_in_low_case():
    c = cfg()
    c.scope.deduct[0].weight = pr.Assumption.model_validate(a("d", 0.5, 1.0, 1.0))
    r = pr.restate(c, FACTS, MARKET, fx_rate=1.0)
    scope_low, scope_high = r.steps[2].factor[0], r.steps[2].factor[2]
    assert scope_low == pytest.approx((400 + 50 - 50) / 1000)    # include low, deduct high
    assert scope_high == pytest.approx((400 + 150 - 25) / 1000)  # include high, deduct low


def test_grade_is_worst_input():
    r = pr.restate(cfg(), FACTS, MARKET, fx_rate=1.0)
    assert r.grade == "C" and r.steps[0].grade == "A" and r.steps[1].grade == "A"


def test_missing_fact_raises_clear_error():
    with pytest.raises(KeyError, match="edgar fetch"):
        pr.restate(cfg(revenue={"period": "FY2030"}), FACTS, MARKET, fx_rate=1.0)


@pytest.mark.parametrize("bad", [
    a("x", 0.5, 0.4, 0.6),            # base < low
    a("x", 0.5, 0.5, 0.5, "C"),       # C must be a range
    a("x", 0.1, 0.2, 0.3, "D"),       # unknown grade
    a("x", 14, 14, 14, "A", unit="%", quote="Academic and government were 15% of revenue"),  # quote lacks value
])
def test_assumption_validation(bad):
    with pytest.raises(ValidationError):
        pr.Assumption.model_validate(bad)


def test_assumption_quote_ok():
    pr.Assumption.model_validate(a("x", 14, 14, 14, "A", unit="%", quote="Academic and government were 14% of revenue"))


def test_check_evidence():
    c = cfg()
    assert pr.check_evidence(c, {"https://sec/x.htm": "We serve academic  and government markets."}) == []
    assert pr.check_evidence(c, {"https://sec/x.htm": "nothing"})[0].startswith("peer_x.acad")
    assert "not cached" in pr.check_evidence(c, {})[0]


def test_fact_rows_are_derived_c_and_pass_rule_1():
    from src.facts_store import check_rows

    r = pr.restate(cfg(), FACTS, MARKET, fx_rate=1.25)
    res = pr.RunResult([r], [cfg()], MARKET)
    rows = res.fact_rows()
    assert {x.metric for x in rows} == {"ds_academia_revenue_low", "ds_academia_revenue_base", "ds_academia_revenue_high"}
    assert {x.region for x in rows} == {"dach", "total"} and all(x.grade == "C" for x in rows)
    check_rows(rows)
    keys = {x.key for x in res.assumption_rows()}
    assert {"peer_x.ds", "peer_x.w_mixed", "market.c_dach"} <= keys


def test_fx_average_from_cache(tmp_path):
    import json
    (tmp_path / "USD_2025.json").write_text(json.dumps(
        {"source_url": "https://ecb/x", "monthly": {f"2025-{m:02d}": 1.0 + m / 100 for m in range(1, 13)}}))
    rate, src = pr.fx_units_per_eur("USD", 2025, cache_dir=tmp_path)
    assert rate == pytest.approx(1.065) and src == "https://ecb/x"
    with pytest.raises(FileNotFoundError):
        pr.fx_units_per_eur("USD", 2024, cache_dir=tmp_path)


@pytest.mark.parametrize("peer", pr.PEERS)
def test_real_configs_load(peer):
    c = pr.load_peer(peer)
    assert c.peer == peer and c.period == "FY2025"
    assert c.academia is not None and c.academia.eg_share.grade == "A"          # disclosed end-market share
    assert c.academia.eg_share.quote and c.academia.eg_share.source_file and c.academia.eg_share.page
    assert all(x.grade == "C" for x in (c.ds_share, c.merck_resale_share, c.academia.concentration))
    m = pr.load_market()
    assert set(m.clusters) == {"dach", "france_benelux", "uk_ireland_nordics"}
    assert sum(x.base for x in m.clusters.values()) < 1.0



def _composite_cfg():
    d = cfg().model_dump()
    d.pop("academia_share")
    d["academia"] = {"eg_share": a("eg", 0.10, 0.10, 0.10, "A", quote="Education & government (10%)", quote_value=10),
                     "concentration": a("conc", 1.2, 1.5, 1.8), "emea_factor": a("emea", 0.9, 1.0, 1.1),
                     "hospital_addon": a("hosp", 0.02, 0.05, 0.08)}
    return pr.PeerConfig.model_validate(d)


def test_composite_academia_share():
    r = pr.restate(_composite_cfg(), FACTS, MARKET, fx_rate=1.0)
    lo, base, hi = r.steps[4].factor
    assert base == pytest.approx(0.10 * 1.5 * 1.0 + 0.05)
    assert lo == pytest.approx(0.10 * 1.2 * 0.9 + 0.02) and hi == pytest.approx(0.10 * 1.8 * 1.1 + 0.08)
    assert r.steps[4].grade == "C"


def test_exactly_one_academia_definition():
    d = _composite_cfg().model_dump()
    d["academia_share"] = a("acad", 0.2, 0.25, 0.3)
    with pytest.raises(ValidationError):
        pr.PeerConfig.model_validate(d)


def test_monte_carlo_percentiles_inside_bounds_and_reproducible():
    r1 = pr.restate(_composite_cfg(), FACTS, MARKET, fx_rate=1.0, n_draws=5000, seed=7)
    r2 = pr.restate(_composite_cfg(), FACTS, MARKET, fx_rate=1.0, n_draws=5000, seed=7)
    p10, p50, p90 = r1.mc["total"]
    lo, base, hi = r1.total
    assert lo <= p10 <= p50 <= p90 <= hi
    assert (p90 - p10) < (hi - lo) * 0.7          # the point of the exercise: narrower than min-max
    assert r1.mc == r2.mc
    assert r1.mc["merck_total"][1] < p50


def test_triangular_constant_when_disclosed():
    import numpy as np
    x = pr.triangular(pr.Assumption.model_validate(a("k", 0.14, 0.14, 0.14, "A")), 10, np.random.default_rng(0))
    assert (x == 0.14).all()


def test_sensitivity_ranks_widest_input_first():
    d = _composite_cfg().model_dump()
    d["ds_share"] = a("ds", 0.1, 0.5, 0.9)          # very wide -> must dominate
    r = pr.restate(pr.PeerConfig.model_validate(d), FACTS, MARKET, fx_rate=1.0, n_draws=4000, seed=1)
    assert r.sensitivity[0].key == "peer_x.ds"
    assert r.sensitivity[0].swing > 0 and 0 < r.sensitivity[0].var_share <= 1
    assert all(s.key != "peer_x.eg" for s in r.sensitivity)   # constant (disclosed) inputs have no swing


def test_check_source_quotes(tmp_path):
    from tests.test_pdf_extract import make_pdf
    pdf_path = make_pdf(tmp_path / "us_x_2025_annual-report.pdf", ["Cover", "Academic and Government 14 percent"])
    good = pr.Assumption.model_validate(a("ag", 0.14, 0.14, 0.14, "A", quote="Academic and Government 14 percent",
                                          quote_value=14, source_file=pdf_path.name, page=2))
    bad_page = good.model_copy(update={"page": 1})
    assert pr.check_source_quotes([good], tmp_path) == []
    assert "not on page 1" in pr.check_source_quotes([bad_page], tmp_path)[0]


def test_dach_crosscheck_flags_implausible_estimates():
    import numpy as np
    from src.facts_store import FactRow

    def f(entity, segment, value, unit="EUR_k"):
        return FactRow("manual_pdf", "x.pdf", TS, entity, "FY2025", "revenue", value, unit, None, segment, "q", "A")
    facts = [f("vwr_international_gmbh", "Chemikalien", 142_598), f("vwr_international_gmbh", "Labortechnik und Sonstiges", 416_692),
             f("vwr_international_gmbh", None, 559_290), f("vwr_international_gmbh", "Deutschland", 376_328)]
    market = pr.Market.model_validate({"clusters": {"dach": a("c", 0.3, 0.35, 0.4)}, "fx": {},
        "dach_anchor": {"germany_share_of_dach": a("g", 0.72, 0.77, 0.82), "fisher_academic_share": a("f", 0.2, 0.3, 0.4),
                        "vwr_lab_consumables_share": a("v", 0.45, 0.55, 0.65), "vwr_academic_share": a("va", 0.12, 0.17, 0.22)}})

    def res(level):
        r = pr.Restatement("avantor", "Avantor", "FY2025", [], (0, 0, 0), {}, {}, 1.0, "", "C")
        r.mc_draws = {"dach": np.full(2000, level)}
        return r
    ok = pr.dach_crosscheck([res(50.0)], facts, market, seed=1)[0]
    too_high = pr.dach_crosscheck([res(500.0)], facts, market, seed=1)[0]
    assert ok.plausible and not too_high.plausible and "IMPLAUSIBLE" in too_high.verdict
