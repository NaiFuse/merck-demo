"""OECD MSTI (Main Science and Technology Indicators): UK academic R&D expenditure and researchers.

    uv run python -m src.connectors.oecd_msti fetch [--refresh]   # network -> data/raw/oecd/ (no DB)
    uv run python -m src.connectors.oecd_msti show

Bridges the Eurostat gap: Eurostat has no UK data after 2019, and its UK HERD predates ONS's 2018 revision.
Official OECD SDMX REST API (no key), one CSV request. Values stay in national currency (GBP_m) as published;
src/models/research_base.py converts at the ECB annual average rate. DB writes happen in src/facts_store.py.
"""

from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime

from src.connectors.eurostat import load_config
from src.connectors.http_cache import PoliteClient, cached_get
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "oecd"
SCALE_SUFFIX = {"0": "", "3": "_k", "6": "_m", "9": "_bn"}


def request(cfg: dict) -> tuple[str, dict]:
    o = cfg["oecd_msti"]
    areas = "+".join(o["countries"].values())
    measures = "+".join(o["measures"])
    units = "+".join(sorted({m["unit"] for m in o["measures"].values()}))
    return f"{o['base_url']}/{areas}.A.{measures}.{units}..", {"startPeriod": str(cfg["first_year"]), "format": "csvfile"}


def parse_csv(text: str, cfg: dict, url: str, retrieved_at: datetime) -> list[FactRow]:
    o = cfg["oecd_msti"]
    iso = {v: k for k, v in o["countries"].items()}
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        m = o["measures"].get(r["MEASURE"])
        if not m or r["UNIT_MEASURE"] != m["unit"] or not r["OBS_VALUE"] or r["REF_AREA"] not in iso:
            continue
        value = float(r["OBS_VALUE"])
        if m["unit"] == "XDC":
            unit = "GBP" + SCALE_SUFFIX[r.get("UNIT_MULT") or "0"]   # only GB is configured; XDC = GBP
        else:
            unit = "count"
        flag = r.get("OBS_STATUS") or None
        quote = (f"OECD MSTI [REF_AREA={r['REF_AREA']}, MEASURE={r['MEASURE']}, UNIT_MEASURE={r['UNIT_MEASURE']}, "
                 f"PRICE_BASE={r.get('PRICE_BASE')}, UNIT_MULT={r.get('UNIT_MULT')}, TIME_PERIOD={r['TIME_PERIOD']}]: "
                 f"{r['OBS_VALUE']}" + (f" (status {flag})" if flag and flag != "A" else ""))
        c = iso[r["REF_AREA"]]
        out.append(FactRow("oecd_msti", url, retrieved_at, c, r["TIME_PERIOD"], m["metric"], value, unit, c,
                           m["sector"], quote, "A", {"flag": flag if flag != "A" else None}))
    return out


def load_all(online: bool = False, refresh: bool = False, cfg: dict | None = None) -> list[FactRow]:
    cfg = cfg or load_config()
    url, params = request(cfg)
    client = PoliteClient(min_interval=1.0) if online else None
    try:
        body, meta = cached_get(client, url, params, CACHE_DIR, ".csv", refresh=refresh)
    finally:
        if client:
            client.close()
    return parse_csv(body.decode("utf-8-sig"), cfg, meta["url"], meta["retrieved_at"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--refresh", action="store_true")
    sub.add_parser("show")
    args = ap.parse_args(argv)
    facts = load_all(online=args.cmd == "fetch", refresh=getattr(args, "refresh", False))
    print(f"{len(facts)} OECD MSTI facts")
    for r in sorted(facts, key=lambda r: (r.metric, r.segment, r.period)):
        print(f"  {r.entity} {r.metric:<15} {r.segment} {r.period} {r.value:>12,.1f} {r.unit}  {r.extra.get('flag') or ''}")


if __name__ == "__main__":
    main()
