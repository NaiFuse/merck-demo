from datetime import date

import pandas as pd
import pytest

from src.models.tender_report import (COVERAGE_NOTES, group_shares, win_shares, in_scope, is_instrument_bundle, period_label,
                                      value_by_country_source)

START, END = date(2023, 11, 18), date(2026, 11, 18)


def test_period_labels_mark_partial_years():
    assert period_label(date(2023, 12, 1), START, END) == "2023 partial (from 18 Nov)"
    assert period_label(date(2024, 6, 1), START, END) == "2024"
    assert period_label(date(2025, 6, 1), START, END) == "2025"
    assert period_label(date(2026, 3, 1), START, END) == "2026 YTD (to 18 Nov)"


def _df():
    rows = [
        ("n1", "thermo_fisher", 60.0, None, "core", False, "2024", date(2024, 1, 1)),
        ("n2", "merck", 30.0, None, "core", True, "2024", date(2024, 2, 1)),
        ("n3", "merck", 99.0, "lab_water", "core", False, "2024", date(2024, 3, 1)),
        ("n4", "ivd", 500.0, None, "core", False, "2024", date(2024, 4, 1)),
        ("n5", "other", 10.0, None, "core", False, "2024", date(2024, 5, 1)),
        ("n6", "avantor", 1000.0, None, "broad", False, "2024", date(2024, 6, 1)),
    ]
    df = pd.DataFrame(rows, columns=["notice_id", "group", "value_eur", "scope_note", "tier", "is_framework",
                                     "period", "publication_date"])
    df["instrument_bundle"] = False
    return df


def test_in_scope_drops_ivd_lab_water_and_other_tier():
    assert list(in_scope(_df())["notice_id"]) == ["n1", "n2", "n5"]


def test_group_shares_with_and_without_frameworks():
    d = in_scope(_df())
    s = group_shares(d)
    assert s.loc["2024", "Thermo Fisher"] == pytest.approx(60.0)
    assert s.loc["2024", "Merck"] == pytest.approx(30.0)
    assert "IVD / diagnostics (out of scope)" not in s.columns
    s2 = group_shares(d[~d["is_framework"]])
    assert s2.loc["2024", "Merck"] == 0
    assert s2.loc["2024", "Thermo Fisher"] == pytest.approx(60 / 70 * 100)


@pytest.mark.parametrize("title,flag", [
    ("Fourniture de réactifs, consommables et maintenances pour automates", True),
    ("Laboratorioautomaation ja analytiikan kokonaisuus", True),
    ("Lieferung von Laborverbrauchsmaterial", False),
    ("Laboratory and Research Reagents and Kits", False),
    (None, False),
])
def test_instrument_bundle_flag(title, flag):
    assert is_instrument_bundle(title) is flag


def test_uk_breakdown_by_country_and_source_and_coverage_notes():
    d = pd.DataFrame({
        "cluster": ["UK, Ireland & Nordics"] * 3 + ["DACH"],
        "country": ["GB", "IE", "SE", "DE"], "source": ["uk_fts", "ted", "ted", "ted"],
        "period": ["2024", "2024", "2025", "2024"], "value_eur": [5e6, 2e6, 1e6, 9e6],
        "publication_date": [date(2024, 1, 1), date(2024, 1, 2), date(2025, 1, 1), date(2024, 1, 3)],
    })
    t = value_by_country_source(d, "UK, Ireland & Nordics")
    assert t.loc[("GB", "uk_fts"), "2024"] == 5.0
    assert ("DE", "ted") not in t.index
    assert any("2021" in n for n in COVERAGE_NOTES) and any("Contracts Finder" in n for n in COVERAGE_NOTES)


def test_default_view_excludes_analyser_bundles_unless_requested():
    df = _df()
    df.loc[df["notice_id"] == "n1", "instrument_bundle"] = True
    assert list(in_scope(df)["notice_id"]) == ["n2", "n5"]
    assert list(in_scope(df, include_bundles=True)["notice_id"]) == ["n1", "n2", "n5"]


def test_win_shares_by_value_and_count():
    d = in_scope(_df())  # thermo 60 (1 award), merck 30 (1), other 10 (1)
    w = win_shares(d)
    assert w.loc["2024", ("Thermo Fisher", "value %")] == pytest.approx(60.0)
    assert w.loc["2024", ("Thermo Fisher", "awards %")] == pytest.approx(100 / 3)
    assert w.loc["2024", ("total", "awards")] == 3


def test_uk_coverage_notes_and_markers():
    from src.models.tender_report import mark_uk_coverage, periods_beyond_uk, uk_coverage_notes

    cov = {"start": date(2023, 11, 1), "end": date(2026, 1, 31), "first_month": "2023-11", "last_month": "2026-01",
           "missing_months": [], "low_scan_months": ["2024-02"], "median_scanned": 855}
    notes = uk_coverage_notes(cov, START, END)
    assert "18 Nov 2023 .. 31 Jan 2026" in notes[0]
    assert any("MISSING, not zero" in n for n in notes) and any("2024-02" in n for n in notes)
    assert "not covered" in uk_coverage_notes(None, START, END)[0]

    d = pd.DataFrame({"period": ["2025", "2026 YTD (to 18 Nov)"], "publication_date": [date(2025, 6, 1), date(2026, 5, 1)]})
    beyond = periods_beyond_uk(d, date(2026, 1, 31))
    assert beyond == {"2026 YTD (to 18 Nov)"}
    idx = pd.MultiIndex.from_product([["DACH", "UK, Ireland & Nordics"], ["2025", "2026 YTD (to 18 Nov)"]],
                                     names=["cluster", "period"])
    t = mark_uk_coverage(pd.DataFrame({"total EUR m": range(4)}, index=idx), beyond, date(2026, 1, 31),
                         "UK, Ireland & Nordics")
    assert list(t["UK coverage"]) == ["", "", "", "UK to 31 Jan 2026"]


def _awards(rows):
    return pd.DataFrame(rows, columns=["notice_id", "winner_name", "alignment"])


def test_framework_like_rule_scotland_excel_shape():
    from src.models.tender_report import framework_like_notices

    se = [("FTS-052813-2025", f"Supplier {i}", "split_equal") for i in range(20)]       # Scotland Excel shape
    four = [("n4", f"S{i}", "split_equal") for i in range(4)]                           # < 5 suppliers
    mixed = [("nm", f"S{i}", "split_equal") for i in range(5)] + [("nm", "X", "single")]  # not all equal splits
    novalue = [("nv", f"S{i}", "no_value") for i in range(6)]                          # no value at all
    assert framework_like_notices(_awards(se + four + mixed + novalue)) == {"FTS-052813-2025"}


def test_in_scope_applies_academia_filter_and_removal_table():
    from src.models.tender_report import filter_removal

    df = _df()
    df["academia"] = [True, True, True, True, False, True]
    df["source"] = "ted"
    df["buyer_category"] = ["university", "hospital", "hospital", "hospital", "municipality", "research"]
    d = in_scope(df)
    assert list(d["notice_id"]) == ["n1", "n2"]
    r = filter_removal(in_scope(df, academia=False), d)
    assert r.loc[("ted", "municipality"), "removed_eur_m"] == pytest.approx(10 / 1e6)
    assert r.loc[("ted", "ALL removed"), "notices"] == 1


def test_cluster_group_shares_dach_merck():
    from src.models.tender_report import cluster_group_shares

    d = pd.DataFrame({
        "cluster": ["DACH", "DACH", "DACH", "France & Benelux"],
        "group": ["merck", "other", "merck", "merck"],
        "value_eur": [30.0, 70.0, 100.0, 999.0],
        "is_framework": [False, False, True, False],
        "notice_id": ["a", "b", "c", "d"],
        "period": ["2025"] * 4,
        "publication_date": [date(2025, 1, 1)] * 4,
    })
    t = cluster_group_shares(d, "DACH", "merck")
    assert t.loc["2025", ("all awards", "value %")] == pytest.approx(130 / 200 * 100)
    assert t.loc["2025", ("excl. frameworks", "value %")] == pytest.approx(30.0)
    assert t.loc["2025", ("all awards", "awards %")] == pytest.approx(200 / 3)



@pytest.mark.parametrize("title,flag", [
    ("Blodprovtagningsartiklar 2025, SLL412", True),
    ("HSE 21963 National Contract for Phlebotomy and Blood Collection", True),
    ("Provtagningsmaterial inkl.vacuumrör", True),
    ("Material zur Blutentnahme", True),
    ("Fourniture de dispositifs de prélèvement sanguin", True),
    ("Laboratory Consumables & Chemicals", False),
    ("Fourniture de sachets de prélèvement pour le laboratoire", False),   # lab sampling bags, not blood
])
def test_clinical_consumables_flag(title, flag):
    from src.models.tender_report import is_clinical_consumables
    assert is_clinical_consumables(title) is flag


def test_in_scope_drops_clinical_consumables():
    df = _df()
    df["clinical_consumables"] = [False, True, False, False, False, False]
    assert list(in_scope(df)["notice_id"]) == ["n1", "n5"]
