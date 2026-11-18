"""SEC EDGAR: Thermo Fisher and Avantor revenue by segment, sub-segment, product type and geography.

    uv run python -m src.connectors.edgar fetch [--refresh]   # network -> data/raw/edgar/ cache (no DB)
    uv run python -m src.connectors.edgar show [--period FY2025]

Official APIs only (https://www.sec.gov/os/accessing-edgar-data): data.sec.gov submissions + companyfacts JSON and
the filing archive (index.json, XBRL instance *_htm.xml, primary 10-K document). User-Agent comes from SEC_USER_AGENT
as SEC requires; requests are kept well below the 10 req/s fair-access limit.

companyfacts has no dimensional data, so segment/geo/product revenue is read from each filing's XBRL instance.
Every fact keeps the filing URL and an exact rendering of the XBRL fact as its quote (rule 1). DB writes are not done
here: facts go to src/facts_store.py in a separate short batch (rule 7).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.extract.numbers import quote_contains_value
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT, get_settings

PEERS = {
    "thermo_fisher": "0000097745",
    "avantor": "0001722482",
}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{name}"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "edgar"
FORMS = ("10-K", "10-Q")
SINCE = "2023-01-01"          # report dates from FY2023 on (Avantor's last geographic segments are in FY2023)
MIN_INTERVAL_S = 0.25         # 4 req/s

XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDI = "{http://xbrl.org/2006/xbrldi}"
REVENUE_CONCEPTS = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"}
GEO_AXES = {"srt:StatementGeographicalAxis", "us-gaap:GeographicDistributionAxis"}
KEEP_AXES = GEO_AXES | {
    "us-gaap:StatementBusinessSegmentsAxis",
    "us-gaap:SubsegmentsAxis",
    "us-gaap:SubsegmentsConsolidationItemsAxis",
    "srt:ProductOrServiceAxis",
    "srt:ConsolidationItemsAxis",
}
AXIS_LABEL = {
    "us-gaap:StatementBusinessSegmentsAxis": "segment",
    "us-gaap:SubsegmentsAxis": "subsegment",
    "us-gaap:SubsegmentsConsolidationItemsAxis": "subsegment_consolidation",
    "srt:ProductOrServiceAxis": "product",
    "srt:ConsolidationItemsAxis": "consolidation",
}


# ---------------------------------------------------------------- HTTP + cache

class EdgarClient:
    def __init__(self, user_agent: str | None = None, client: httpx.Client | None = None,
                 min_interval: float = MIN_INTERVAL_S, retries: int = 5, sleep=time.sleep):
        ua = user_agent if user_agent is not None else get_settings().sec_user_agent
        if not ua:
            raise RuntimeError("SEC_USER_AGENT is not set in .env (SEC requires a descriptive User-Agent)")
        self._client = client or httpx.Client(timeout=120, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
        if client is not None:
            self._client.headers["User-Agent"] = ua
        self._min_interval = min_interval
        self._retries = retries
        self._sleep = sleep
        self._last = 0.0

    def close(self) -> None:
        self._client.close()

    def get(self, url: str) -> bytes:
        delay = 2.0
        for attempt in range(self._retries):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._client.get(url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        raise _RetryAfter(float(retry_after))
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.content
            except _RetryAfter as ra:
                if attempt == self._retries - 1:
                    raise
                self._sleep(ra.seconds)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == self._retries - 1:
                    raise
                self._sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")


class _RetryAfter(Exception):
    def __init__(self, seconds: float):
        self.seconds = seconds


def cache_path(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    u = urlparse(url)
    return cache_dir / u.netloc / u.path.lstrip("/")


def cached_get(client: EdgarClient | None, url: str, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> tuple[bytes, datetime]:
    """Bytes from cache, else from the network (then cached). Returns (content, retrieved_at = file mtime)."""
    path = cache_path(url, cache_dir)
    if path.exists() and not refresh:
        return path.read_bytes(), datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
    if client is None:
        raise FileNotFoundError(f"not cached and offline: {url}")
    content = client.get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content, datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------- filings

@dataclass(frozen=True)
class Filing:
    entity: str
    cik: str
    accession: str        # with dashes
    form: str
    report_date: str
    primary_doc: str

    @property
    def acc_nodash(self) -> str:
        return self.accession.replace("-", "")

    def url(self, name: str) -> str:
        return ARCHIVE_URL.format(cik_int=int(self.cik), acc=self.acc_nodash, name=name)

    @property
    def instance_guess(self) -> str:
        return self.primary_doc.rsplit(".", 1)[0] + "_htm.xml"


def select_filings(entity: str, submissions: dict, forms=FORMS, since: str = SINCE) -> list[Filing]:
    rec = submissions["filings"]["recent"]
    cik = str(submissions["cik"]).zfill(10)
    out = []
    for i, form in enumerate(rec["form"]):
        if form in forms and rec["reportDate"][i] >= since:
            out.append(Filing(entity, cik, rec["accessionNumber"][i], form, rec["reportDate"][i], rec["primaryDocument"][i]))
    return sorted(out, key=lambda f: f.report_date)


def instance_name(index_json: dict, fallback: str) -> str:
    names = [i["name"] for i in index_json.get("directory", {}).get("item", [])]
    hits = [n for n in names if n.endswith("_htm.xml")]
    return hits[0] if hits else fallback


# ---------------------------------------------------------------- XBRL

@dataclass(frozen=True)
class XbrlFact:
    concept: str          # prefixed, e.g. 'us-gaap:Revenues'
    context_id: str
    start: date
    end: date
    dims: tuple[tuple[str, str], ...]
    raw: str              # value text exactly as in the instance
    unit: str
    decimals: str | None

    @property
    def value(self) -> float:
        return float(self.raw)


def _prefix_map(xml_bytes: bytes) -> dict[str, str]:
    """namespace URI -> prefix as declared in the instance."""
    out = {}
    for _, (prefix, uri) in ET.iterparse(io.BytesIO(xml_bytes), events=["start-ns"]):
        out.setdefault(uri, prefix)
    return out


def parse_instance(xml_bytes: bytes, concepts: set[str] = REVENUE_CONCEPTS) -> list[XbrlFact]:
    prefixes = _prefix_map(xml_bytes)
    root = ET.fromstring(xml_bytes)
    ctx = {}
    for c in root.iter(XBRLI + "context"):
        p = c.find(XBRLI + "period")
        s, e = p.findtext(XBRLI + "startDate"), p.findtext(XBRLI + "endDate")
        if not (s and e):
            continue  # instant contexts are not revenue
        dims = tuple(sorted((m.get("dimension"), m.text.strip()) for m in c.iter(XBRLDI + "explicitMember")))
        typed = any(True for _ in c.iter(XBRLDI + "typedMember"))
        ctx[c.get("id")] = (date.fromisoformat(s.strip()), date.fromisoformat(e.strip()), dims, typed)
    units = {}
    for u in root.iter(XBRLI + "unit"):
        m = u.find(XBRLI + "measure")
        units[u.get("id")] = (m.text or "").split(":")[-1] if m is not None else u.get("id")
    out = []
    for el in root:
        if not el.tag.startswith("{"):
            continue
        uri, local = el.tag[1:].split("}")
        if local not in concepts or el.get("contextRef") not in ctx or el.text is None:
            continue
        s, e, dims, typed = ctx[el.get("contextRef")]
        if typed:
            continue
        out.append(XbrlFact(f"{prefixes.get(uri, uri)}:{local}", el.get("contextRef"), s, e, dims,
                            el.text.strip(), units.get(el.get("unitRef"), el.get("unitRef")), el.get("decimals")))
    return out


def period_label(start: date, end: date) -> str | None:
    """'FY2025' for ~12 months, '2026Q3' for ~3 months, None otherwise (YTD contexts are skipped)."""
    days = (end - start).days
    if 350 <= days <= 380:
        return f"FY{end.year}"
    if 80 <= days <= 100:
        anchor = end - timedelta(days=10)  # 52/53-week quarters can end a few days into the next month
        return f"{anchor.year}Q{(anchor.month - 1) // 3 + 1}"
    return None


def _local(member: str) -> str:
    name = member.split(":")[-1]
    return name[:-6] if name.endswith("Member") else name


def split_dims(dims: tuple[tuple[str, str], ...]) -> tuple[str | None, str | None]:
    """(region, segment) strings for the facts table. region = geo member; segment = other axes 'label=member;...'."""
    region = next((_local(m) for a, m in dims if a in GEO_AXES), None)
    rest = [f"{AXIS_LABEL.get(a, a)}={_local(m)}" for a, m in dims if a not in GEO_AXES]
    return region, (";".join(rest) or None)


def quote_of(f: XbrlFact) -> str:
    dims = ", ".join(f"{a}={m}" for a, m in f.dims) or "none"
    return (f"{f.concept} | context={f.context_id} | {f.start.isoformat()}..{f.end.isoformat()} | "
            f"dims: {dims} | value={f.raw} {f.unit} | decimals={f.decimals}")


def revenue_rows(filing: Filing, facts: list[XbrlFact], source_url: str, retrieved_at: datetime) -> list[FactRow]:
    rows = []
    for f in facts:
        if f.unit != "USD" or any(a not in KEEP_AXES for a, _ in f.dims):
            continue
        period = period_label(f.start, f.end)
        if not period:
            continue
        region, segment = split_dims(f.dims)
        value_m = f.value / 1e6
        quote = quote_of(f)
        if not quote_contains_value(quote, value_m * 1e6, "USD"):
            raise AssertionError(f"quote check failed: {quote}")
        rows.append(FactRow("edgar", source_url, retrieved_at, filing.entity, period, "revenue", value_m, "USD_m",
                            region, segment, quote, "A",
                            {"form": filing.form, "accession": filing.accession, "report_date": filing.report_date}))
    return rows


def latest_per_key(rows: list[FactRow]) -> list[FactRow]:
    """Each 10-K/10-Q repeats prior periods; keep the value from the most recent filing (restated, current segments)."""
    best: dict[tuple, FactRow] = {}
    for r in rows:
        key = (r.entity, r.period, r.metric, r.region, r.segment)
        if key not in best or r.extra["report_date"] > best[key].extra["report_date"]:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r.entity, r.period, r.region or "", r.segment or ""))


def companyfacts_totals(cf: dict) -> dict[str, float]:
    """FY totals (USD m) from companyfacts, for the cross-check against instance values."""
    out = {}
    for concept in REVENUE_CONCEPTS:
        for item in cf.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get("USD", []):
            if "start" not in item:
                continue
            p = period_label(date.fromisoformat(item["start"]), date.fromisoformat(item["end"]))
            if p and p.startswith("FY"):
                out[p] = item["val"] / 1e6
    return out


def cross_check(rows: list[FactRow], totals: dict[str, float], tol: float = 0.5) -> list[str]:
    """Problems where the undimensioned instance revenue disagrees with companyfacts (empty list = all agree)."""
    problems = []
    for r in rows:
        if r.region is None and r.segment is None and r.period in totals and abs(r.value - totals[r.period]) > tol:
            problems.append(f"{r.entity} {r.period}: instance {r.value:,.1f} vs companyfacts {totals[r.period]:,.1f}")
    return problems


def html_text(html: bytes | str) -> str:
    """Visible text of an EDGAR HTML document, whitespace-collapsed (for checking quotes offline)."""
    s = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html
    s = re.sub(r"(?is)<(script|style|ix:header)\b.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", unescape(s)).strip()


# ---------------------------------------------------------------- orchestration

def load_entity(entity: str, client: EdgarClient | None, refresh: bool = False,
                cache_dir: Path = CACHE_DIR) -> tuple[list[FactRow], list[str]]:
    """All revenue rows for one peer (network only if not cached; client=None = offline). Returns (rows, problems)."""
    cik = PEERS[entity]
    subs_raw, _ = cached_get(client, SUBMISSIONS_URL.format(cik=cik), refresh, cache_dir)
    filings = select_filings(entity, json.loads(subs_raw))
    rows: list[FactRow] = []
    for f in filings:
        idx_raw, _ = cached_get(client, f.url("index.json"), refresh, cache_dir)
        name = instance_name(json.loads(idx_raw), f.instance_guess)
        url = f.url(name)
        xml_bytes, retrieved = cached_get(client, url, False, cache_dir)  # filings never change
        rows += revenue_rows(f, parse_instance(xml_bytes), url, retrieved)
        if f.form == "10-K":
            cached_get(client, f.url(f.primary_doc), False, cache_dir)  # narrative text for assumption quotes
    rows = latest_per_key(rows)
    cf_raw, _ = cached_get(client, COMPANYFACTS_URL.format(cik=cik), refresh, cache_dir)
    return rows, cross_check(rows, companyfacts_totals(json.loads(cf_raw)))


def load_all(online: bool = False, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> tuple[list[FactRow], list[str]]:
    client = EdgarClient() if online else None
    try:
        rows, problems = [], []
        for entity in PEERS:
            r, p = load_entity(entity, client, refresh, cache_dir)
            rows += r
            problems += p
        return rows, problems
    finally:
        if client:
            client.close()


def ten_k_texts(entity: str, cache_dir: Path = CACHE_DIR) -> dict[str, str]:
    """{source_url: visible text} for the cached 10-K primary documents of a peer (offline)."""
    out = {}
    subs = cache_path(SUBMISSIONS_URL.format(cik=PEERS[entity]), cache_dir)
    if not subs.exists():
        return out
    for f in select_filings(entity, json.loads(subs.read_bytes()), forms=("10-K",)):
        p = cache_path(f.url(f.primary_doc), cache_dir)
        if p.exists():
            out[f.url(f.primary_doc)] = html_text(p.read_bytes())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--refresh", action="store_true", help="re-download submissions/index/companyfacts")
    s = sub.add_parser("show")
    s.add_argument("--period", default="FY2025")
    args = ap.parse_args(argv)
    rows, problems = load_all(online=args.cmd == "fetch", refresh=getattr(args, "refresh", False))
    if args.cmd == "fetch":
        print(f"{len(rows)} revenue facts cached for {', '.join(PEERS)}")
    period = getattr(args, "period", "FY2025")
    for r in rows:
        if r.period == period and (r.segment is None or "consolidation" not in (r.segment or "")):
            print(f"{r.entity:<14}{r.period:<8}{(r.region or '-'):<20}{(r.segment or '-')[:70]:<72}{r.value:>12,.1f}")
    print("cross-check vs companyfacts:", "OK" if not problems else "; ".join(problems))


if __name__ == "__main__":
    main()
