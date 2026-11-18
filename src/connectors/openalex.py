"""OpenAlex: wet-lab publication counts by country and year (academia-type institutions).

    uv run python -m src.connectors.openalex fetch [--refresh]   # network -> data/raw/openalex/ (no DB)
    uv run python -m src.connectors.openalex show [--year 2025]

Official REST API (https://docs.openalex.org). One grouped request per (year, query): /works filtered to
publication_year, academia-type institution types and a wet-lab topic filter, group_by=authorships.countries.
Full counting: a work counts once for every country among its authors. Two disjoint topic queries per year are
summed in the model, not here. OPENALEX_EMAIL (polite pool) and OPENALEX_API_KEY are optional and never stored
in the cache key, the cached URL or the DB. DB writes happen in src/facts_store.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from src.connectors.eurostat import load_config
from src.connectors.http_cache import PoliteClient, cached_get
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT, get_settings

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "openalex"
GROUP_BY = "authorships.countries"


def params_for(cfg: dict, year: int, query: str) -> dict:
    o = cfg["openalex"]
    filt = ",".join([f"publication_year:{year}",
                     "authorships.institutions.type:" + "|".join(o["institution_types"]),
                     o["queries"][query]])
    return {"filter": filt, "group_by": GROUP_BY, "per_page": "200"}


def secret_params() -> dict:
    s = get_settings()
    out = {}
    if s.openalex_email:
        out["mailto"] = s.openalex_email
    if s.openalex_api_key:
        out["api_key"] = s.openalex_api_key.get_secret_value()
    return out


def parse_groups(d: dict) -> dict[str, int]:
    """ISO-2 country -> work count. Groups look like {'key': 'https://openalex.org/countries/DE', 'count': n}."""
    out = {}
    for g in d.get("group_by", []):
        code = str(g["key"]).rsplit("/", 1)[-1].upper()
        if len(code) == 2:
            out[code] = int(g["count"])
    return out


def to_facts(counts: dict[str, int], year: int, query: str, url: str, retrieved_at: datetime,
             countries: set[str]) -> list[FactRow]:
    rows = []
    for c, n in sorted(counts.items()):
        if c not in countries:
            continue
        quote = f"OpenAlex /works group_by={GROUP_BY} [{query}, publication_year={year}]: key=countries/{c}, count={n}"
        rows.append(FactRow("openalex", url, retrieved_at, c, str(year), "publications", float(n), "count", c,
                            query, quote, "A"))
    return rows


def load_all(online: bool = False, refresh: bool = False, cfg: dict | None = None) -> list[FactRow]:
    cfg = cfg or load_config()
    o = cfg["openalex"]
    countries = set(cfg["europe_countries"])
    client = PoliteClient(min_interval=0.2) if online else None
    secrets = secret_params() if online else {}
    facts = []
    try:
        for year in range(cfg["first_year"], o["last_year"] + 1):
            for query in o["queries"]:
                body, meta = cached_get(client, o["base_url"], params_for(cfg, year, query), CACHE_DIR, ".json",
                                        refresh=refresh, secret_params=secrets)
                d = json.loads(body)
                if d.get("meta", {}).get("groups_count", 0) >= 200 and len(parse_groups(d)) < 30:
                    raise RuntimeError(f"OpenAlex group_by truncated for {year} {query}")
                facts += to_facts(parse_groups(d), year, query, meta["url"], meta["retrieved_at"], countries)
    finally:
        if client:
            client.close()
    return facts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--refresh", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("--year", default="2025")
    args = ap.parse_args(argv)
    facts = load_all(online=args.cmd == "fetch", refresh=getattr(args, "refresh", False))
    print(f"{len(facts)} OpenAlex facts")
    year = getattr(args, "year", "2025")
    tot: dict[str, float] = {}
    for r in facts:
        if r.period == year:
            tot[r.entity] = tot.get(r.entity, 0) + r.value
    for c, n in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {c} {n:>10,.0f}")


if __name__ == "__main__":
    main()
