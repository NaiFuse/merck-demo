from datetime import datetime

from src.connectors.ecb_fx import parse_ecb_csv, to_eur

CSV = """KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE
EXR.M.SEK.EUR.SP00.A,M,SEK,EUR,SP00,A,2025-01,11.4882
EXR.M.SEK.EUR.SP00.A,M,SEK,EUR,SP00,A,2025-02,
EXR.M.DKK.EUR.SP00.A,M,DKK,EUR,SP00,A,2025-01,7.4605
"""


def test_parse_ecb_csv_skips_empty_values():
    rates = parse_ecb_csv(CSV, "https://data-api.ecb.europa.eu/x", datetime(2026, 11, 18))
    assert [(r.currency, r.month, r.units_per_eur) for r in rates] == [("SEK", "2025-01", 11.4882), ("DKK", "2025-01", 7.4605)]


def test_to_eur():
    rates = {("SEK", "2025-01"): 11.0}
    assert to_eur(110.0, "SEK", "2025-01", rates) == 10.0
    assert to_eur(5.0, "EUR", None, rates) == 5.0
    assert to_eur(5.0, "NOK", "2025-01", rates) is None
    assert to_eur(None, "EUR", "2025-01", rates) is None
