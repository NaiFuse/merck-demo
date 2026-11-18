"""CORDIS: EU framework-programme grants (H2020, Horizon Europe) to academia-type participants by country and year.

    uv run python -m src.connectors.cordis fetch [--refresh]   # network -> data/raw/cordis/*.zip (no DB)
    uv run python -m src.connectors.cordis show

Source: the official CORDIS bulk CSV downloads (data.europa.eu, EC reuse notice) – no query API exists, and these
are published for reuse, so this is not scraping. organization.csv (one row per participation) is joined to
project.csv on projectID; year = year of the project's ecSignatureDate. Kept: activityType in
config/research_sources.yaml (HES higher education, REC research organisations). Aggregate =
sum(netEcContribution) per (programme, country, year, activityType), plain Python. The aggregation is cached
next to the zip (keyed by zip size + mtime). DB writes happen in src/facts_store.py.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.connectors.eurostat import FROM_EUROSTAT, load_config
from src.connectors.http_cache import read_meta
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cordis"
csv.field_size_limit(10**8)   # project objectives are long


def zip_path(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / url.rsplit("/", 1)[1]


def download(url: str, cache_dir: Path = CACHE_DIR, refresh: bool = False) -> Path:
    """Stream the zip to the cache (+ .meta.json). Network only, no DB."""
    import httpx

    path = zip_path(url, cache_dir)
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url, timeout=600, follow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(path)
    _write_meta(path, url, datetime.now(timezone.utc).replace(tzinfo=None))
    return path


def _write_meta(path: Path, url: str, ts: datetime) -> None:
    Path(str(path) + ".meta.json").write_text(json.dumps({"url": url, "retrieved_at": ts.isoformat()}, indent=1))


def meta_for(path: Path, url: str) -> dict:
    """Provenance of a cached zip. A zip downloaded by hand gets its file mtime as retrieval time."""
    if not Path(str(path) + ".meta.json").exists():
        _write_meta(path, url, datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None))
    return read_meta(path)


def _amount(s: str) -> float:
    s = (s or "").strip()
    return float(s.replace(",", ".")) if s else 0.0


def _rows(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""), delimiter=";")


def aggregate(path: Path, activity_types: list[str]) -> dict[tuple[str, int, str], dict]:
    """(country ISO-2, signature year, activityType) -> {'net': EUR, 'n': participations}."""
    with zipfile.ZipFile(path) as zf:
        year = {}
        for p in _rows(zf, "project.csv"):
            d = (p.get("ecSignatureDate") or "")[:4]
            if d.isdigit():
                year[p["id"]] = int(d)
        agg: dict = defaultdict(lambda: {"net": 0.0, "n": 0})
        for o in _rows(zf, "organization.csv"):
            if o["activityType"] not in activity_types or o["projectID"] not in year:
                continue
            c = FROM_EUROSTAT.get(o["country"], o["country"])
            a = agg[(c, year[o["projectID"]], o["activityType"])]
            a["net"] += _amount(o["netEcContribution"])
            a["n"] += 1
    return dict(agg)


def cached_aggregate(path: Path, activity_types: list[str]) -> dict[tuple[str, int, str], dict]:
    st = path.stat()
    key = {"size": st.st_size, "mtime": st.st_mtime, "activity_types": activity_types}
    cache = path.with_suffix(".aggregate.json")
    if cache.exists():
        d = json.loads(cache.read_text())
        if d["key"] == key:
            return {(c, int(y), t): v for c, y, t, v in d["rows"]}
    agg = aggregate(path, activity_types)
    cache.write_text(json.dumps({"key": key, "rows": [[c, y, t, v] for (c, y, t), v in sorted(agg.items())]}))
    return agg


def to_facts(agg: dict, programme: str, url: str, retrieved_at: datetime, countries: set[str]) -> list[FactRow]:
    rows = []
    for (c, y, t), v in sorted(agg.items()):
        if c not in countries:
            continue
        net = round(v["net"], 2)
        quote = (f"CORDIS {programme} bulk CSV: sum(organization.netEcContribution) over {v['n']} participations "
                 f"with country={c}, activityType={t}, project ecSignatureDate year={y} = {net:.2f} EUR")
        rows.append(FactRow("cordis", f"{url}#organization.csv+project.csv", retrieved_at, c, str(y),
                            "eu_grant_net_contribution", net, "EUR", c, t, quote, "A",
                            {"programme": programme, "participations": v["n"]}))
    return rows


def load_all(online: bool = False, refresh: bool = False, cfg: dict | None = None) -> list[FactRow]:
    cfg = cfg or load_config()
    c = cfg["cordis"]
    facts = []
    for programme, url in c["files"].items():
        path = download(url, refresh=refresh) if online else zip_path(url)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing – run `python -m src.connectors.cordis fetch`")
        meta = meta_for(path, url)
        agg = cached_aggregate(path, c["activity_types"])
        facts += to_facts(agg, programme, meta["url"], meta["retrieved_at"], set(cfg["europe_countries"]))
    return facts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--refresh", action="store_true")
    sub.add_parser("show")
    args = ap.parse_args(argv)
    facts = load_all(online=args.cmd == "fetch", refresh=getattr(args, "refresh", False))
    print(f"{len(facts)} CORDIS facts")
    by: dict = defaultdict(float)
    for r in facts:
        by[(r.entity, r.extra["programme"])] += r.value
    for (c, p), v in sorted(by.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {c} {p:<8} {v / 1e6:>10,.1f} EUR m (HES+REC, all years)")


if __name__ == "__main__":
    main()
