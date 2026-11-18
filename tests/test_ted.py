"""TED connector: query building, slicing, parsing, provenance check, batched writes (no network)."""

from datetime import date, datetime

import pytest

from pathlib import Path

from src.connectors import ted, ted_xml, tender_store
from src.connectors.ted import Slice
from src.db import connect

TS = datetime(2026, 11, 18, 12, 0)
C3 = {"DEU", "FRA", "SWE"}

# Shapes copied from real TED v3 responses (2026-11-18), trimmed.
SINGLE = {
    "publication-number": "8729-2025", "notice-type": "can-standard", "publication-date": "2025-01-08+01:00",
    "buyer-name": {"deu": ["Deutsches Zentrum für Neurodegenerative Erkrankungen e.V."]},
    "buyer-country": ["DEU"], "classification-cpv": ["33696500", "33696500"],
    "winner-name": {"deu": ["Quanterix Inc"]}, "winner-identifier": ["NL825580900"],
    "result-value-notice": "300066.64", "total-value": 300066.64, "total-value-cur": ["EUR"],
    "links": {"xml": {"MUL": "https://ted.europa.eu/en/notice/8729-2025/xml"}},
}
MULTI_ALIGNED = {
    "publication-number": "4234-2025", "notice-type": "can-standard", "publication-date": "2025-01-03+02:00",
    "buyer-name": {"fra": ["Université X"]}, "buyer-country": ["FRA"], "buyer-legal-type": ["body-pl"],
    "classification-cpv": ["33696500", "38437110"],
    "winner-name": {"fra": ["MILTENYI BIOTEC S.A.S.", "STEMCELL TECHNOLOGIES"]},
    "winner-identifier": ["419 828 900 00037", "417 977 139 00050"],
    "winner-decision-date": ["2024-11-20+02:00", "2024-11-20+02:00"],
    "result-value-lot": ["920000", "216000"], "result-value-cur-lot": ["EUR"],
}
MULTI_UNALIGNED = {
    "publication-number": "1-2025", "notice-type": "can-standard", "publication-date": "2025-02-01+01:00",
    "buyer-name": {"swe": ["Lunds universitet"]}, "buyer-country": ["SWE"], "classification-cpv": ["33696300"],
    "winner-name": {"swe": ["VWR International AB", "Merck Life Science AB", "Thermo Fisher Scientific AB"]},
    "winner-identifier": ["a", "b"],
    "result-value-notice": "900000", "result-value-cur-notice": ["SEK"],
    "result-value-lot": ["100000", "800000"],
}


def test_query_contains_all_filters():
    q = ted.build_query(["33696300", "33696500"], Slice("DEU", date(2024, 1, 1), date(2024, 12, 31)))
    assert q == ("classification-cpv IN (33696300 33696500) AND buyer-country IN (DEU) "
                 "AND notice-type IN (can-standard can-social) "
                 "AND publication-date>=20240101 AND publication-date<=20241231")


def test_lookback_and_year_slices():
    start = ted.lookback_start(date(2026, 11, 18))
    assert start == date(2023, 11, 18)
    assert ted.lookback_start(date(2024, 3, 31), 1) == date(2024, 2, 29)
    sl = ted.year_slices(["DEU"], start, date(2026, 11, 18))
    assert [(s.start, s.end) for s in sl] == [
        (date(2023, 11, 18), date(2023, 12, 31)), (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 11, 18))]


def test_plan_splits_slices_over_cap():
    class FakeClient:
        def count(self, q):  # 2 notices per day
            s = q.split("publication-date>=")[1][:8]
            e = q.split("publication-date<=")[1][:8]
            return 2 * ((datetime.strptime(e, "%Y%m%d") - datetime.strptime(s, "%Y%m%d")).days + 1)

    out = ted.plan(FakeClient(), ["1"], [Slice("DEU", date(2024, 1, 1), date(2024, 12, 31))], cap=200)
    assert all(n <= 200 for _, n in out)
    assert sum(n for _, n in out) == 2 * 366
    assert out[0][0].start == date(2024, 1, 1) and out[-1][0].end == date(2024, 12, 31)


def test_cluster_countries_cover_taxonomy():
    assert set(ted.cluster_countries_iso3()) == {"DEU", "AUT", "CHE", "FRA", "BEL", "NLD", "LUX",
                                                 "GBR", "IRL", "SWE", "NOR", "DNK", "FIN", "ISL"}


def test_cpv_config_is_valid():
    cfg = ted.load_cpv_config()
    assert all(len(c) == 8 and c.isdigit() for c in cfg["include"])
    assert {v["tier"] for v in cfg["include"].values()} <= {"core", "broad"}
    assert "33696200" in cfg["exclude"]  # blood-testing reagents = diagnostics


def test_parse_single_winner_with_provenance():
    t, a = ted.parse_notice(SINGLE, "Q", TS, {"33696500"}, C3)
    assert t["country"] == "DE" and t["value"] == 300066.64 and t["currency"] == "EUR"
    assert t["value_field"] == "result-value-notice"
    assert t["cpv"] == "33696500" and t["cpv_all"] == "33696500"
    assert t["source_url"] == "https://ted.europa.eu/en/notice/-/detail/8729-2025"
    assert t["api_query"] == "Q" and "links" not in t["raw_json"] and "300066.64" in t["raw_json"]
    assert t["award_date"] == date(2025, 1, 8)  # falls back to publication date
    assert len(a) == 1
    assert {k: a[0][k] for k in ("seq", "winner_name", "winner_id", "value", "currency", "alignment", "grade")} == dict(
        seq=0, winner_name="Quanterix Inc", winner_id="NL825580900", value=300066.64, currency="EUR",
        alignment="single", grade="A")
    assert a[0]["evidence"] == "result-value-notice=300066.64"
    assert t["is_framework"] is False


def test_parse_lot_aligned_uses_lot_values():
    t, a = ted.parse_notice(MULTI_ALIGNED, "Q", TS, {"33696500", "38437110"}, C3)
    assert t["value"] == 1136000 and t["value_field"] == "result-value-lot(sum)"
    assert t["award_date"] == date(2024, 11, 20)
    assert [(x["winner_name"], x["value"], x["alignment"], x["grade"]) for x in a] == [
        ("MILTENYI BIOTEC S.A.S.", 920000.0, "lot", "B"), ("STEMCELL TECHNOLOGIES", 216000.0, "lot", "B")]


def test_parse_unaligned_splits_equally_grade_c():
    _, a = ted.parse_notice(MULTI_UNALIGNED, "Q", TS, {"33696300"}, C3)
    assert [x["alignment"] for x in a] == ["split_equal"] * 3
    assert all(x["grade"] == "C" and x["value"] == 300000 and x["currency"] == "SEK" for x in a)
    assert all(x["winner_id"] is None for x in a)  # 2 ids for 3 winners: not attributable


def test_verify_value_rejects_mismatch():
    with pytest.raises(ValueError, match="not found"):
        ted.verify_value(SINGLE, "result-value-notice", 1.0)


def test_apply_fx_converts_with_award_month():
    t, a = ted.parse_notice(MULTI_UNALIGNED, "Q", TS, {"33696300"}, C3)
    ted.apply_fx([t], a, {("SEK", "2025-02"): 11.25})
    assert t["value_eur"] == pytest.approx(80000)
    assert a[0]["value_eur"] == pytest.approx(300000 / 11.25)


def test_save_upserts_in_batches_with_short_connections(tmp_path, monkeypatch):
    rows = [ted.parse_notice(n, "Q", TS, {"33696500", "33696300", "38437110"}, C3)
            for n in (SINGLE, MULTI_ALIGNED, MULTI_UNALIGNED)]
    tenders = [r[0] for r in rows]
    awards = [x for r in rows for x in r[1]]
    ted.apply_fx(tenders, awards, {("SEK", "2025-02"): 11.25})

    opened = []
    real_connect = tender_store.connect
    monkeypatch.setattr(tender_store, "connect", lambda p=None: opened.append(p) or real_connect(p))
    path = tmp_path / "t.duckdb"
    ted.save(tenders, awards, [], path, batch_size=2)
    ted.save(tenders, awards, [], path, batch_size=2)  # idempotent re-run
    assert len(opened) == 6  # 1 init + 2 batches, twice
    with connect(path) as con:
        assert con.execute("SELECT count(*) FROM tenders").fetchone()[0] == 3
        assert con.execute("SELECT count(*) FROM tender_awards").fetchone()[0] == 6
        assert con.execute("SELECT count(*) FROM tenders WHERE source_url IS NULL OR raw_json IS NULL").fetchone()[0] == 0


def test_run_fetch_never_holds_db_during_network(tmp_path, monkeypatch):
    """Rule 7: no DB connection is open while TED or ECB is called."""
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

    def net(*a, **k):
        state["network_while_open"] += state["open"] > 0

    class FakeClient:
        def __init__(self, *a, **k): pass
        def close(self): pass
        def count(self, q):
            net(); return 1 if "DEU" in q and "2025" in q else 0
        def search(self, q, page=1, **k):
            net(); return {"notices": [SINGLE] if page == 1 else []}

    monkeypatch.setattr(ted, "TedClient", FakeClient)
    monkeypatch.setattr(ted.ecb_fx, "fetch_monthly_rates", lambda *a, **k: net() or [])
    monkeypatch.setattr(ted.ted_xml, "fetch_xml", lambda ids, **k: net() or {})
    out = ted.run_fetch(today=date(2026, 11, 18), db_path=tmp_path / "f.duckdb")
    assert out["notices"] == 1
    assert state["network_while_open"] == 0


# ---------------------------------------------------------------- milestone 2 additions

FIXTURE_XML = (Path(__file__).parent / "fixtures" / "ted_notice_min.xml").read_text(encoding="utf-8")


def test_text_fields_as_plain_strings_are_not_split_into_characters():
    raw = dict(SINGLE, **{"title-proc": {"deu": "Lieferung von Laborverbrauchsmaterial"}})
    t, _ = ted.parse_notice(raw, "Q", TS, {"33696500"}, C3)
    assert t["title"] == "Lieferung von Laborverbrauchsmaterial"


def test_framework_flag():
    assert ted.is_framework(dict(MULTI_ALIGNED, **{"framework-agreement-lot": ["fa-wo-rc", "fa-wo-rc"]}))
    assert ted.is_framework(dict(SINGLE, **{"result-framework-maximum-value-notice": "1818000"}))
    assert not ted.is_framework(dict(SINGLE, **{"framework-agreement-lot": ["none"]}))
    assert not ted.is_framework(SINGLE)


def test_xml_parser_resolves_lots_and_splits_multi_winner_lot():
    res = ted_xml.parse_notice_xml(FIXTURE_XML)
    got = [(a.lot_id, a.winner_name, a.value, a.value_kind, a.grade) for a in res.awards]
    assert got == [
        ("LOT-0001", "Merck Life Science GmbH", 48000.50, "payable", "A"),
        ("LOT-0002", "VWR International GmbH", 45000.0, "framework_max_split", "C"),  # group lead
        ("LOT-0002", "Carl Roth GmbH", 45000.0, "framework_max_split", "C"),
    ]
    assert "PayableAmount=48000.50" in res.awards[0].evidence  # quote contains the number
    assert res.awards[0].winner_id == "DE111"
    assert not res.fully_resolved
    assert res.lot_texts["LOT-0001"].startswith("Reinstwasseranlage Milli-Q")


def test_select_for_xml_takes_top_bc_core_non_ivd():
    def tender(nid, v, cpv="33696500"):
        return {"notice_id": nid, "value_eur": v, "cpv": cpv}

    def award(nid, grade, name="Carl Roth GmbH"):
        return {"notice_id": nid, "grade": grade, "winner_name": name}

    tenders = [tender("a", 100), tender("b", 900), tender("c", 500), tender("d", 800, "24310000"), tender("e", 700)]
    awards = [award("a", "C"), award("b", "A"), award("c", "B"), award("d", "C"), award("e", "C", "Roche Diagnostics")]
    assert ted.select_for_xml(tenders, awards, {"33696500"}, n=5) == ["c", "a"]
    assert ted.select_for_xml(tenders, awards, {"33696500"}, n=1) == ["c"]


def test_lab_water_tagging_on_merck_xml_lot():
    t, _ = ted.parse_notice(SINGLE, "Q", TS, {"33696500"}, C3)
    res = ted_xml.parse_notice_xml(FIXTURE_XML)
    awards = ted.xml_awards(t["notice_id"], "https://x/xml", res)
    t["_lot_texts"] = list(res.lot_texts.values())
    ted.apply_scope_flags([t], awards)
    assert [a["scope_note"] for a in awards] == ["lab_water", None, None]  # only Merck is checked


def test_xml_reconciliation_rejects_repeated_ceilings_and_zero_values():
    def res(values, names=None):
        names = names or ["A"] * len(values)
        return ted_xml.XmlResult([ted_xml.XmlAward("L", "T", n, None, v, "EUR", "framework_max", "A", "")
                                  for v, n in zip(values, names)])

    assert ted.xml_reconciles(res([60.0, 40.0]), 100.0) == (True, "")
    ok, why = ted.xml_reconciles(res([100.0] * 147), 100.0)  # buyer repeated notice ceiling on every lot
    assert not ok and "x147" in why
    assert not ted.xml_reconciles(res([0.0, 0.0]), 100.0)[0]
    assert not ted.xml_reconciles(res([50.0, 50.0], ["A", None]), 100.0)[0]
    assert ted.xml_reconciles(res([50.0]), None)[0]  # no notice value: nothing to reconcile against


def test_xml_reference_with_scheme_name_does_not_overwrite_definition():
    xml = FIXTURE_XML.replace(
        "</efac:NoticeResult>",
        '<efac:SettledContract><efac:LotTender><cbc:ID schemeName="tender">TEN-0001</cbc:ID></efac:LotTender>'
        "</efac:SettledContract></efac:NoticeResult>")
    a = ted_xml.parse_notice_xml(xml).awards[0]
    assert a.winner_name == "Merck Life Science GmbH" and a.value == 48000.50


def test_xml_zero_payable_amount_falls_back_to_lot_maximum():
    xml = FIXTURE_XML.replace(">48000.50<", ">0<")
    a = ted_xml.parse_notice_xml(xml).awards[0]
    assert (a.value, a.value_kind) == (50000.0, "framework_max")


def test_lot_pairing_requires_lot_values_to_reconcile_with_notice_value():
    raw = dict(MULTI_ALIGNED, **{"result-value-notice": "1000000"})  # lots sum to 1,136,000: within 25%
    assert {a["alignment"] for a in ted.parse_notice(raw, "Q", TS, {"33696500"}, C3)[1]} == {"lot"}
    raw = dict(MULTI_ALIGNED, **{"result-value-notice": "100000", "result-value-lot": ["100000", "100000"]})
    _, a = ted.parse_notice(raw, "Q", TS, {"33696500"}, C3)  # ceiling repeated per lot: x2
    assert [(x["alignment"], x["value"], x["grade"]) for x in a] == [("split_equal", 50000, "C")] * 2
