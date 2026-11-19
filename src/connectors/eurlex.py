"""EU procurement thresholds from the official Commission delegated regulations (Publications Office CELLAR API).

    uv run python -m src.connectors.eurlex fetch     # downloads the acts listed in config/eurlex.yaml (XHTML)
    uv run python -m src.connectors.eurlex show      # offline: verified facts from the cache

CELLAR content negotiation: GET http://publications.europa.eu/resource/celex/<CELEX> with Accept: application/xhtml+xml
and Accept-Language: eng returns the Official Journal text (the eur-lex.europa.eu web pages are not used: they sit
behind a bot challenge, and CELLAR is the documented machine interface). Cached under data/raw/eurlex/ with a
.meta.json sidecar (URL, retrieval time). Each configured quote must appear verbatim in the act's text and contain
its value; failures raise.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import yaml

from src.connectors.http_cache import PoliteClient, cached_get
from src.extract.numbers import quote_contains_value
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "eurlex"
CONFIG = PROJECT_ROOT / "config" / "eurlex.yaml"
CELLAR = "http://publications.europa.eu/resource/celex/{celex}"
HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng", "User-Agent": "merck-ds-demo/0.1 (research)"}


def text_of(xhtml: bytes) -> str:
    t = re.sub(r"<[^>]+>", " ", xhtml.decode("utf-8", errors="replace"))
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ").replace(" ", " ").replace(" ", " ")).strip()


def load_all(online: bool = False, config: Path = CONFIG, cache_dir: Path = CACHE_DIR) -> list[FactRow]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    client = PoliteClient(min_interval=1.0, headers=HEADERS) if online else None
    rows = []
    try:
        for act in cfg["acts"]:
            body, meta = cached_get(client, CELLAR.format(celex=act["celex"]), None, cache_dir, ".xhtml")
            text = _norm(text_of(body))
            for f in act["facts"]:
                q = _norm(f["quote"])
                if q not in text:
                    raise ValueError(f"{act['celex']}: quote not in the act's text: {f['quote']!r}")
                if not quote_contains_value(q, f["value"], "EUR"):
                    raise ValueError(f"{act['celex']}: quote does not contain {f['value']}")
                for year in act["applies"]:
                    rows.append(FactRow(
                        "eurlex", act["eli"], meta["retrieved_at"].replace(tzinfo=None),
                        "eu", str(year), f["metric"], float(f["value"]), "EUR", "EU", "Directive 2014/24/EU Art. 4",
                        f"{act['title']} (CELEX {act['celex']}, fetched {meta['url']}): \"{f['quote']}\"", "A"))
    finally:
        if client:
            client.close()
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["fetch", "show"])
    args = ap.parse_args(argv)
    for r in load_all(online=args.cmd == "fetch"):
        print(f"{r.period} {r.metric:<44} EUR {r.value:>9,.0f}  [{r.grade}]  {r.url_or_path}")


if __name__ == "__main__":
    main()
