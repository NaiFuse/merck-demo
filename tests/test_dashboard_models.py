"""Share bridge, Merck DS estimate, scenario overrides and the app's range formatting – offline, no DB."""

import numpy as np
import pytest

from src.models import merck_ds, share_bridge
from src.models import scenario as sc
from src.models.peer_restatement import Assumption, Market


def a(key, low, base, high, grade="C", **kw):
    return Assumption(key=key, low=low, base=base, high=high, grade=grade, source="test", **kw)


# ---------------------------------------------------------------- share bridge

W = {"acad": 0.4, "pharma": 0.4, "ind": 0.2}
V = {"acad": 0.3, "pharma": 0.5, "ind": 0.2}
G = {"acad": 0.05, "pharma": 0.03, "ind": 0.02}


def test_bridge_identity_with_segment_growth():
    g_ds_s = {"acad": 0.02, "pharma": 0.04, "ind": 0.01}
    out = share_bridge.decompose(W, V, G, g_ds_s=g_ds_s)
    assert out["g_m"] == pytest.approx(0.3 * 0.05 + 0.5 * 0.03 + 0.2 * 0.02)
    assert out["mix"] == pytest.approx(0.1 * 0.05 - 0.1 * 0.03)
    assert out["g_ds"] - out["g_m"] == pytest.approx(out["mix"] + out["share"])
    assert out["share_s"]["acad"] == pytest.approx(0.4 * (0.02 - 0.05))


def test_bridge_residual_share_matches_segment_version():
    g_ds_s = {"acad": 0.02, "pharma": 0.04, "ind": 0.01}
    full = share_bridge.decompose(W, V, G, g_ds_s=g_ds_s)
    resid = share_bridge.decompose(W, V, G, g_ds=full["g_ds"])
    assert resid["share"] == pytest.approx(full["share"])
    with pytest.raises(ValueError):
        share_bridge.decompose(W, V, G)


def test_bridge_works_on_draws():
    n = 1000
    w = {k: np.full(n, v) for k, v in W.items()}
    out = share_bridge.decompose(w, V, G, g_ds=np.linspace(0, 0.04, n))
    assert out["share"].shape == (n,)
    assert np.allclose(out["g_ds"] - out["g_m"], out["mix"] + out["share"])


def test_bridge_build_weights_sum_to_one():
    cfg = merck_ds.load_config()
    n = 2000
    br = share_bridge.build(cfg.bridge, np.full(n, 0.38), np.full(n, 0.045), n, seed=1)
    assert np.allclose(sum(br.w.values()), 1.0)
    assert np.allclose(sum(br.v.values()), 1.0)
    assert br.grade == "C"


# ---------------------------------------------------------------- Merck DS estimate

def _market():
    return Market(fx={}, clusters={"x": a("cs_x", 0.2, 0.2, 0.2, "B"), "y": a("cs_y", 0.3, 0.3, 0.3, "B")})


def test_merck_config_quotes_and_grades():
    cfg = merck_ds.load_config()
    assert all(x.grade == "A" and x.quote and x.source_file for x in cfg.disclosed.values())
    assert all(x.grade == "C" and x.low < x.high for x in cfg.estimate.values())
    assert cfg.default_emea_base() == pytest.approx(2900 * 0.33)


def test_ds_estimate_scales_with_slider_and_ranges_shares():
    cfg = merck_ds.load_config()
    n = 5000
    supply = {"x": np.full(n, 500.0), "y": np.full(n, 500.0), "total": np.full(n, 1000.0)}
    demand = {"x": np.full(n, 1000.0), "y": np.full(n, 1000.0), "total": np.full(n, 2000.0)}
    methods = {"A": supply, "B": demand}
    rng = {k: (500.0, None, 1000.0, "range") for k in ("x", "y")} | {"total": (1000.0, None, 2000.0, "range")}
    d0 = merck_ds.estimate(cfg, _market(), methods, rng, n, seed=7)
    d1 = merck_ds.estimate(cfg, _market(), methods, rng, n, seed=7, emea_target=d0.emea_default / 2)
    assert np.allclose(d1.academia["total"], d0.academia["total"] / 2)
    assert np.allclose(d0.academia["total"], d0.academia["x"] + d0.academia["y"])
    s = d0.shares["total"]
    p10, _, p90 = merck_ds.pct(d0.academia["total"])
    assert s.base is None and s.low == pytest.approx(p10 / 2000) and s.high == pytest.approx(p90 / 1000)
    assert set(s.vs) == {"A", "B"} and s.vs_a[1] > s.vs_b[1]
    pub = merck_ds.estimate(cfg, _market(), methods, rng | {"total": (1000.0, 1500.0, 2000.0, "published")},
                            n, seed=7).shares["total"]
    assert pub.base is not None and pub.low < pub.base < pub.high


# ---------------------------------------------------------------- scenario overrides

def test_moved_shift_and_scale_keep_width_and_stay_c():
    x = a("s", 0.55, 0.65, 0.75)
    m = sc.moved(x, 0.70, "shift")
    assert m.base == 0.70 and m.low == pytest.approx(0.60) and m.high == pytest.approx(0.80) and m.grade == "C"
    assert sc.moved(x, 0.95, "shift").high == pytest.approx(0.99)        # share clamp
    i = sc.moved(a("i", 0.010, 0.018, 0.035), 0.009, "scale")
    assert (i.low, i.high) == pytest.approx((0.005, 0.0175))
    fixed = sc.moved(a("f", 1.0, 1.0, 1.0, "A"), 2.0, "shift")
    assert fixed.grade == "C" and fixed.low < fixed.base < fixed.high   # rule 3


def test_replace_assumption_in_models_and_dicts():
    m = Market(fx={}, clusters={"x": a("cs_x", 0.1, 0.2, 0.3)},
               supply={"others_share": {"key": "others_share", "low": 0.5, "base": 0.6, "high": 0.7, "grade": "C",
                                        "source": "s"}})
    out = sc.replace_assumption(m, "others_share", lambda z: sc.moved(z, 0.65, "shift"))
    assert out.supply["others_share"]["base"] == 0.65
    assert m.supply["others_share"]["base"] == 0.6                       # original untouched
    assert sc.find_assumption(out, "cs_x").base == 0.2
    assert sc.find_assumption(out, "nope") is None


def test_knobs_point_at_existing_assumptions():
    from src.models.peer_restatement import load_market, load_peer

    owners = {"market": load_market(), "merck_ds": merck_ds.load_config(),
              "thermo_fisher": load_peer("thermo_fisher"), "avantor": load_peer("avantor")}
    for k in sc.KNOBS:
        if k.uid == sc.EMEA_REVENUE:
            continue
        owner, key = k.uid.split(".", 1)
        found = sc.find_assumption(owners[owner], key)
        assert found is not None, k.uid
        assert k.lo <= found.base <= k.hi, k.uid


# ---------------------------------------------------------------- money-unit quote check

def test_assumption_quote_check_money_unit():
    ok = a("r", 2900, 2900, 2900, "A", unit="EUR_m", quote="Discovery Solutions ~€2.9 bn1")
    assert ok.base == 2900
    with pytest.raises(ValueError):
        a("r", 3100, 3100, 3100, "A", unit="EUR_m", quote="Discovery Solutions ~€2.9 bn1")


# ---------------------------------------------------------------- rule 3 in the UI formatter

def test_figure_shows_c_grade_as_range():
    from src.app.ui import figure

    assert figure(100, 150, 200, "C") == "€100–200m"
    assert figure(100, 150, 200, "B") == "€150m (€100–200m)"
    assert figure(0.05, None, 0.2, "C", "pct") == "5–20%"
    assert "–" in figure(0.05, 0.1, 0.2, "C", "pct")


# ---------------------------------------------------------------- reconciliation

def test_reconcile_range_is_a_b_overlap_with_floor_check_and_anchor_band():
    from src.models import market_model as mm
    from src.models.reconciliation import Check

    n = 4000
    r = np.random.default_rng(0)
    a = mm.SupplyCell("total", 2025, {}, r.normal(1000, 100, n), "C")
    b = mm.DemandCell("total", 2025, 0, 0, 0, 0, r.normal(1250, 200, n), "C")
    anchor = Check("merck_anchor", "x", "total", r.normal(3000, 100, n))
    floor = Check("de_floor", "y", "total", r.normal(400, 20, n), lower_only=True)
    c = mm.reconcile(a, b, 0.15, [anchor, floor])
    assert c.status == "range" and c.grade == "C" and c.base is None and c.members == ["A", "B"]
    assert c.low == pytest.approx(b.pct[0]) and c.high == pytest.approx(a.pct[2])       # max P10 .. min P90
    assert c.floor_check == "ok"
    g = c.bands["merck_anchor"]
    assert g["gap_eur"] == pytest.approx(anchor.pct[0] - c.high) and g["gap_eur"] > 0
    assert g["gap_pct"] == pytest.approx(g["gap_eur"] / c.high)
    # the anchor never moves the range
    assert (mm.reconcile(a, b, 0.15).low, mm.reconcile(a, b, 0.15).high) == (c.low, c.high)
    high_floor = Check("de_floor", "y", "total", r.normal(1200, 10, n), lower_only=True)
    assert mm.reconcile(a, b, 0.15, [high_floor]).floor_check == "floor above range"
    far = mm.DemandCell("total", 2025, 0, 0, 0, 0, r.normal(1700, 100, n), "C")         # no A/B overlap
    c2 = mm.reconcile(a, far, 0.15)
    assert c2.status == "range_no_overlap" and c2.low == pytest.approx(a.pct[0]) and c2.high == pytest.approx(far.pct[2])


def test_band_gap_signs():
    from src.models.market_model import band_gap

    assert band_gap(100, 200, (250, 300, 350))["gap_eur"] == 50
    assert band_gap(100, 200, (20, 50, 80))["gap_eur"] == -20
    assert band_gap(100, 200, (150, 250, 350))["gap_eur"] == 0


def test_bridge_lsd_view_scales_market_growth():
    cfg = merck_ds.load_config()
    n = 4000
    lsd = Assumption.model_validate(cfg.anchor["market_cagr"])
    br = share_bridge.build(cfg.bridge, np.full(n, 0.38), np.full(n, 0.045), n, seed=3, lsd=lsd)
    m, l = br.views["model"], br.views["merck_lsd"]
    assert np.allclose(m["g_ds"], l["g_ds"])                                   # same DS growth
    assert 0.01 <= np.percentile(l["g_m"], 1) and np.percentile(l["g_m"], 99) <= 0.03
    assert np.allclose(l["g_ds"] - l["g_m"], l["mix"] + l["share"])            # identity in both views
    assert np.median(l["share"]) > np.median(m["share"])                       # lower market growth -> less share loss
    assert br.draws is m


def test_tri_median():
    from src.models.reconciliation import tri_median

    assert tri_median(a("s", 0.0, 0.5, 1.0)) == pytest.approx(0.5)
    x = np.random.default_rng(1).triangular(0.7, 0.82, 0.87, 200000)
    assert tri_median(a("o", 0.7, 0.82, 0.87)) == pytest.approx(np.median(x), abs=2e-3)


def test_others_share_views_from_awards():
    import pandas as pd
    from src.models import others_share

    rows = []
    for g, v, fw in [("thermo_fisher", 20, False), ("avantor", 10, False), ("merck", 5, False),
                     ("distributor", 15, True), ("other", 50, False)]:
        rows.append({"group": g, "value_eur": v * 1e6, "is_framework": fw, "period": "2025", "tier": "core",
                     "scope_note": None, "clinical_consumables": False, "academia": True, "instrument_bundle": False})
    df = pd.DataFrame(rows)
    vs = others_share.views(df)
    v = {(x.measure, x.frameworks, x.distributors): x.others for x in vs}
    assert v[("value", True, "others")] == pytest.approx(0.70)          # 1 - 30/100
    assert v[("value", True, "pro rata")] == pytest.approx(1 - 30 / 85)
    assert v[("value", False, "others")] == pytest.approx(1 - 30 / 85)  # distributor award was the framework
    assert v[("count", True, "others")] == pytest.approx(0.6)
    d, _ = others_share.derive(df)
    assert d.grade == "B" and d.low == pytest.approx(min(v.values()), abs=1e-4)


def test_manual_facts_verified():
    from src.extract import manual_facts

    rows, problems = manual_facts.load_all()
    assert not problems
    f = next(r for r in rows if r.metric == "addressable_market_size")
    assert (f.value, f.unit, f.grade, f.period) == (80.0, "EUR_bn", "A", "FY2025")
    assert "p.14" in f.quote and "Discovery3" in f.quote


def test_manual_facts_reject_wrong_value():
    from src.extract.manual_facts import verify

    item = {"page": 1, "label": "Discovery3 label", "value_fragment": "~€80 bn", "value": 35, "unit": "EUR_bn",
            "quotes": ["Discovery3 label", "~€35 bn ~€80 bn ~€105 bn"]}
    pages = ["Discovery3 label ~€35 bn ~€80 bn ~€105 bn"]
    assert any("does not contain" in p for p in verify(item, pages))
    assert verify({**item, "value": 80}, pages) == []
