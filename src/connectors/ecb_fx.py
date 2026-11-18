"""ECB euro reference exchange rates (monthly averages) from the official ECB Data Portal API.

Series EXR.M.<CUR>.EUR.SP00.A = units of <CUR> per 1 EUR, monthly average. No key needed.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/M.{curs}.EUR.SP00.A"


@dataclass(frozen=True)
class FxRate:
    currency: str
    month: str  # 'YYYY-MM'
    units_per_eur: float
    source_url: str
    retrieved_at: datetime


def parse_ecb_csv(text: str, source_url: str, retrieved_at: datetime) -> list[FxRate]:
    rows = csv.DictReader(io.StringIO(text))
    out = []
    for r in rows:
        if not r.get("OBS_VALUE"):
            continue
        out.append(FxRate(r["CURRENCY"], r["TIME_PERIOD"], float(r["OBS_VALUE"]), source_url, retrieved_at))
    return out


def fetch_monthly_rates(currencies: set[str], start_month: str, client: httpx.Client | None = None) -> list[FxRate]:
    """Network only – no DB access. Returns [] for an empty currency set."""
    curs = sorted(c for c in currencies if c and c != "EUR")
    if not curs:
        return []
    url = ECB_URL.format(curs="+".join(curs))
    params = {"startPeriod": start_month, "format": "csvdata"}
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
    finally:
        if own:
            client.close()
    return parse_ecb_csv(resp.text, str(resp.url), datetime.now(timezone.utc).replace(tzinfo=None))


def to_eur(value: float | None, currency: str | None, month: str | None, rates: dict[tuple[str, str], float]) -> float | None:
    """Convert with the monthly rate; None when value, currency or rate is missing."""
    if value is None or not currency:
        return None
    if currency == "EUR":
        return value
    rate = rates.get((currency, month or ""))
    return value / rate if rate else None
