"""Eurostat: academic R&D expenditure and researcher headcount by country and sector of performance.

    uv run python -m src.connectors.eurostat fetch [--refresh]   # network -> data/raw/eurostat/ (no DB)
    uv run python -m src.connectors.eurostat show [--year 2024]

Official dissemination API (JSON-stat 2.0, no key): https://wikis.ec.europa.eu/display/EUROSTATHELP/API+Statistics
Datasets and filters come from config/research_sources.yaml:
  rd_e_gerdtot   GERD by sector of performance, MIO_EUR       (sectors HES, GOV, PNP)
  rd_p_persocc   R&D personnel/researchers, headcount         (researchers, sectors HES, GOV)
One request per dataset. Every fact keeps the exact request URL and a rendering of the JSON-stat cell (with its
Eurostat status flag) as its quote. DB writes happen in src/facts_store.py (rule 7).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from src.connectors.http_cache import PoliteClient, cached_get
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "research_sources.yaml"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "eurostat"
TO_EUROSTAT = {"GB": "UK", "GR": "EL"}
FROM_EUROSTAT = {v: k for k, v in TO_EUROSTAT.items()}
METRIC_UNIT = {"rd_expenditure": "EUR_m", "researchers": "count"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Obs:
    dataset: str
    dims: tuple[tuple[str, str], ...]   # non-time dimensions, e.g. (('sectperf','HES'), ('unit','MIO_EUR'), ('geo','DE'))
    time: str
    value: float
    flag: str | None                    # Eurostat status: e estimated, p provisional, b break, d definition differs, u low reliability

    def dim(self, name: str) -> str:
        return dict(self.dims)[name]


def query_params(filters: dict, countries: list[str], since: int) -> list[tuple[str, str]]:
    params = [("format", "JSON"), ("lang", "EN"), ("sinceTimePeriod", str(since))]
    for k, v in filters.items():
        for x in v if isinstance(v, list) else [v]:
            params.append((k, str(x)))
    params += [("geo", TO_EUROSTAT.get(c, c)) for c in countries]
    return params


def parse_jsonstat(d: dict, dataset: str) -> list[Obs]:
    """Flatten a JSON-stat 2.0 dataset into observations (row-major index over d['id'] / d['size'])."""
    ids, size = d["id"], d["size"]
    cats = {k: sorted(d["dimension"][k]["category"]["index"].items(), key=lambda kv: kv[1]) for k in ids}
    codes = {k: [c for c, _ in cats[k]] for k in ids}
    status = d.get("status") or {}
    out = []
    for flat, value in d.get("value", {}).items():
        if value is None:
            continue
        r, coord = int(flat), []
        for s in reversed(size):
            coord.append(r % s)
            r //= s
        coord.reverse()
        c = {k: codes[k][i] for k, i in zip(ids, coord)}
        dims = tuple((k, c[k]) for k in ids if k not in ("time", "freq"))
        out.append(Obs(dataset, dims, c["time"], float(value), status.get(flat) or None))
    return out


def fetch(name: str, cfg: dict, client: PoliteClient | None, refresh: bool = False) -> tuple[list[Obs], dict]:
    """Observations for one configured query (cfg['eurostat'][name]). client=None -> cache only."""
    q = cfg["eurostat"][name]
    url = f"{cfg['eurostat']['base_url']}/{q['dataset']}"
    params = query_params(q["filters"], cfg["europe_countries"], cfg["first_year"])
    body, meta = cached_get(client, url, params, CACHE_DIR, ".json", refresh=refresh)
    return parse_jsonstat(json.loads(body), q["dataset"]), meta


def quote_for(o: Obs) -> str:
    dims = ", ".join(f"{k}={v}" for k, v in o.dims)
    return f"Eurostat {o.dataset} [{dims}, time={o.time}]: {o.value!r}" + (f" (flag {o.flag})" if o.flag else "")


def to_facts(obs: list[Obs], metric: str, url: str, retrieved_at: datetime) -> list[FactRow]:
    rows = []
    for o in obs:
        country = FROM_EUROSTAT.get(o.dim("geo"), o.dim("geo"))
        rows.append(FactRow("eurostat", url, retrieved_at, country, o.time, metric, o.value, METRIC_UNIT[metric],
                            country, o.dim("sectperf"), quote_for(o), "A", {"flag": o.flag}))
    return rows


def load_all(online: bool = False, refresh: bool = False, cfg: dict | None = None) -> list[FactRow]:
    cfg = cfg or load_config()
    client = PoliteClient(min_interval=1.0) if online else None
    try:
        facts = []
        for metric in ("rd_expenditure", "researchers"):
            obs, meta = fetch(metric, cfg, client, refresh)
            facts += to_facts(obs, metric, meta["url"], meta["retrieved_at"])
        return facts
    finally:
        if client:
            client.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--refresh", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("--year", default="2024")
    args = ap.parse_args(argv)
    facts = load_all(online=args.cmd == "fetch", refresh=getattr(args, "refresh", False))
    print(f"{len(facts)} Eurostat facts")
    year = getattr(args, "year", "2024")
    for metric in METRIC_UNIT:
        rows = sorted((r for r in facts if r.metric == metric and r.period == year), key=lambda r: (r.entity, r.segment))
        print(f"\n{metric} {year} ({METRIC_UNIT[metric]})")
        for r in rows[:40]:
            print(f"  {r.entity} {r.segment:<4} {r.value:>12,.1f}  {r.extra.get('flag') or ''}")


if __name__ == "__main__":
    main()
