from datetime import date

import pandas as pd
import pytest

from src.models.tender_report import group_shares, in_scope, is_instrument_bundle, period_label

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
    return pd.DataFrame(rows, columns=["notice_id", "group", "value_eur", "scope_note", "tier", "is_framework",
                                       "period", "publication_date"])


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
