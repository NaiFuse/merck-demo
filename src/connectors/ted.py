"""TED (Tenders Electronic Daily) contract award notices via the official Search API v3.

    uv run python -m src.connectors.ted cpv     # show the CPV list (config/ted_cpv.yaml)
    uv run python -m src.connectors.ted plan    # count notices per country x year slice, no download
    uv run python -m src.connectors.ted fetch   # download, convert FX, then write to DuckDB in batches

API: POST https://api.ted.europa.eu/v3/notices/search (anonymous, no key).
Per CLAUDE.md rule 7, all network calls finish before the DB write connection is opened, and each
batch opens and closes its own connection.
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import yaml

from src.connectors import ecb_fx, ted_xml
from src.connectors.tender_store import apply_fx, apply_scope_flags, save
from src.models.taxonomy import load_taxonomy
from src.models.winners import IVD, clean_name, winner_group
from src.settings import PROJECT_ROOT

API_URL = "https://api.ted.europa.eu/v3/notices/search"
NOTICE_URL = "https://ted.europa.eu/en/notice/-/detail/{id}"
CPV_CONFIG = PROJECT_ROOT / "config" / "ted_cpv.yaml"

MAX_RESULTS_PER_QUERY = 15_000  # TED hard cap for page-number pagination
PAGE_LIMIT = 250                # max notices per page
MIN_INTERVAL_S = 0.6            # polite spacing between requests
LOOKBACK_MONTHS = 36

# eForms contract award notices. 'can-modif' (contract modifications) is left out to avoid double counting.
NOTICE_TYPES = ("can-standard", "can-social")

FIELDS = [
    "publication-number", "notice-type", "publication-date",
    "buyer-name", "buyer-country", "buyer-legal-type", "buyer-identifier",
    "classification-cpv",
    "winner-name", "winner-identifier", "winner-country", "winner-decision-date", "contract-conclusion-date",
    "result-value-notice", "result-value-cur-notice", "total-value", "total-value-cur",
    "result-value-lot", "result-value-cur-lot", "BT-142-LotResult",
    "framework-agreement-lot", "result-framework-maximum-value-notice",
    "notice-title", "title-proc", "title-lot", "description-lot",
]
XML_TOP_N = 50  # only the top-N grade B/C core notices by value get XML parsing

ISO2_TO_ISO3 = {
    "DE": "DEU", "AT": "AUT", "CH": "CHE", "FR": "FRA", "BE": "BEL", "NL": "NLD", "LU": "LUX",
    "GB": "GBR", "IE": "IRL", "SE": "SWE", "NO": "NOR", "DK": "DNK", "FI": "FIN", "IS": "ISL",
}
ISO3_TO_ISO2 = {v: k for k, v in ISO2_TO_ISO3.items()}


# ---------------------------------------------------------------- config & query building

def load_cpv_config(path: Path = CPV_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overlap = set(cfg["include"]) & set(cfg["exclude"])
    if overlap:
        raise ValueError(f"CPV codes both included and excluded: {sorted(overlap)}")
    return cfg


def cluster_countries_iso3() -> list[str]:
    tax = load_taxonomy()
    return [ISO2_TO_ISO3[c] for cl in tax["clusters"].values() for c in cl["countries"]]


@dataclass(frozen=True)
class Slice:
    country3: str
    start: date
    end: date  # inclusive

    def label(self) -> str:
        return f"{self.country3} {self.start:%Y-%m-%d}..{self.end:%Y-%m-%d}"


def build_query(cpvs: list[str], s: Slice) -> str:
    return (
        f"classification-cpv IN ({' '.join(cpvs)}) AND buyer-country IN ({s.country3}) "
        f"AND notice-type IN ({' '.join(NOTICE_TYPES)}) "
        f"AND publication-date>={s.start:%Y%m%d} AND publication-date<={s.end:%Y%m%d}"
    )


def lookback_start(today: date, months: int = LOOKBACK_MONTHS) -> date:
    y, m0 = divmod(today.year * 12 + today.month - 1 - months, 12)
    return date(y, m0 + 1, min(today.day, calendar.monthrange(y, m0 + 1)[1]))


def year_slices(countries3: list[str], start: date, end: date) -> list[Slice]:
    out = []
    for c in countries3:
        for y in range(start.year, end.year + 1):
            s, e = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
            if s <= e:
                out.append(Slice(c, s, e))
    return out


def split_slice(s: Slice) -> tuple[Slice, Slice]:
    mid = s.start + (s.end - s.start) // 2
    return Slice(s.country3, s.start, mid), Slice(s.country3, mid + timedelta(days=1), s.end)


# ---------------------------------------------------------------- HTTP

class TedClient:
    def __init__(self, client: httpx.Client | None = None, min_interval: float = MIN_INTERVAL_S, retries: int = 5):
        self._client = client or httpx.Client(timeout=60, headers={"User-Agent": "merck-ds-demo/0.1 (research)"})
        self._min_interval = min_interval
        self._retries = retries
        self._last = 0.0

    def close(self) -> None:
        self._client.close()

    def search(self, query: str, page: int = 1, limit: int = PAGE_LIMIT, fields: list[str] | None = None) -> dict:
        payload = {"query": query, "fields": fields or FIELDS, "page": page, "limit": limit,
                   "paginationMode": "PAGE_NUMBER", "scope": "ALL"}
        delay = 2.0
        for attempt in range(self._retries):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._client.post(API_URL, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                if resp.status_code >= 400:  # bad query: retrying won't help
                    raise ValueError(f"TED API {resp.status_code}: {resp.text[:500]}")
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError):
                if attempt == self._retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def count(self, query: str) -> int:
        return int(self.search(query, page=1, limit=1, fields=["publication-number"])["totalNoticeCount"])


def plan(client: TedClient, cpvs: list[str], slices: list[Slice], cap: int = MAX_RESULTS_PER_QUERY) -> list[tuple[Slice, int]]:
    """Count each slice; halve the date range until every slice is under the cap."""
    out, todo = [], list(slices)
    while todo:
        s = todo.pop(0)
        n = client.count(build_query(cpvs, s))
        if n > cap:
            if s.start == s.end:
                raise RuntimeError(f"{s.label()} has {n} notices in a single day – narrow the CPV set")
            todo[:0] = split_slice(s)
        else:
            out.append((s, n))
    return out


def fetch_query(client: TedClient, query: str, expected: int) -> list[dict]:
    notices: list[dict] = []
    page = 1
    while len(notices) < expected:
        if page * PAGE_LIMIT > MAX_RESULTS_PER_QUERY + PAGE_LIMIT:
            raise RuntimeError("paging past the 15,000 cap – slice should have been split")
        batch = client.search(query, page=page)["notices"]
        if not batch:
            break
        notices.extend(batch)
        page += 1
    return notices


# ---------------------------------------------------------------- parsing (pure functions)

def _first_lang(v) -> list[str]:
    """Multilingual fields come as {lang: [values]}; take the first language that has values."""
    if isinstance(v, dict):
        for vals in v.values():
            if vals:
                return [vals] if isinstance(vals, str) else list(vals)  # text fields come as plain strings
        return []
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _date(s: str | None) -> date | None:
    return date.fromisoformat(s[:10]) if s else None


def _float(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def notice_value(raw: dict) -> tuple[float | None, str | None, str | None]:
    """(value, currency, source field). Prefers the notice total, then the sum of lot results."""
    v = _float(raw.get("result-value-notice"))
    if v is not None:
        cur = (_list(raw.get("result-value-cur-notice")) or _list(raw.get("total-value-cur")) or [None])[0]
        return v, cur, "result-value-notice"
    v = _float(raw.get("total-value"))
    if v is not None:
        return v, (_list(raw.get("total-value-cur")) or [None])[0], "total-value"
    lots = [_float(x) for x in _list(raw.get("result-value-lot"))]
    lots = [x for x in lots if x is not None]
    if lots:
        return sum(lots), (_list(raw.get("result-value-cur-lot")) or [None])[0], "result-value-lot(sum)"
    return None, None, None


def award_date(raw: dict) -> date | None:
    for f in ("winner-decision-date", "contract-conclusion-date"):
        ds = [d for d in (_date(x) for x in _list(raw.get(f))) if d]
        if ds:
            return min(ds)
    return _date(raw.get("publication-date"))


def verify_value(raw: dict, field: str | None, value: float | None) -> None:
    """Rule 1: the stored number must be readable back from the raw API row."""
    if value is None:
        return
    if field == "result-value-lot(sum)":
        got = sum(x for x in (_float(v) for v in _list(raw.get("result-value-lot"))) if x is not None)
    else:
        got = _float(raw.get(field))
    if got is None or abs(got - value) > 0.005:
        raise ValueError(f"{raw.get('publication-number')}: value {value} not found in raw field {field}")


def lot_texts(raw: dict) -> list[str]:
    """Per-lot 'title | description' from the search API (positional, may not align with winners)."""
    titles, descs = _first_lang(raw.get("title-lot")), _first_lang(raw.get("description-lot"))
    n = max(len(titles), len(descs))
    return [" | ".join(x for x in (titles[i] if i < len(titles) else None,
                                   descs[i] if i < len(descs) else None) if x) for i in range(n)]


def is_framework(raw: dict) -> bool:
    fa = [x for x in _list(raw.get("framework-agreement-lot")) if x]
    return any(x != "none" for x in fa) or raw.get("result-framework-maximum-value-notice") is not None


def parse_awards(raw: dict, value: float | None, currency: str | None, field: str | None = None) -> list[dict]:
    """One row per winner. Lot-level pairing is used only when names and lot values line up 1:1."""
    names = _first_lang(raw.get("winner-name"))
    ids = _list(raw.get("winner-identifier"))
    raw_lots = _list(raw.get("result-value-lot"))
    lot_vals = [_float(x) for x in raw_lots]
    lot_cur = (_list(raw.get("result-value-cur-lot")) or [currency])[0]
    texts = lot_texts(raw)
    if not names:
        return []

    unique: list[str] = []
    seen: set[str] = set()
    for n in names:
        k = clean_name(n)
        if k not in seen:
            seen.add(k)
            unique.append(n)

    def wid(i: int, pool: list[str]) -> str | None:
        return ids[i] if len(ids) == len(pool) else None

    def row(seq, name, winner_id, val, cur, alignment, grade, evidence, own_text=None):
        return dict(seq=seq, winner_name=name, winner_id=winner_id, value=val, currency=cur,
                    alignment=alignment, grade=grade, evidence=evidence, source_url=None, scope_note=None,
                    _own_text=own_text)

    if value is None:
        return [row(i, n, wid(i, unique), None, None, "no_value", "C", "no award value in search API")
                for i, n in enumerate(unique)]
    if len(unique) == 1:
        own = texts[0] if len(texts) == 1 else None
        return [row(0, unique[0], ids[0] if len(set(ids)) == 1 else None, value, currency, "single", "A",
                    f"{field}={raw.get(field) if field and '(' not in field else value}", own)]
    lots_reconcile = field == "result-value-lot(sum)" or (
        bool(lot_vals) and all(v is not None for v in lot_vals)
        and abs(sum(lot_vals) / value - 1) <= XML_TOLERANCE if value else False)
    if len(names) == len(lot_vals) and all(v is not None for v in lot_vals) and lots_reconcile:
        return [row(i, n, wid(i, names), v, lot_cur, "lot", "B", f"result-value-lot[{i}]={raw_lots[i]} paired by position",
                    texts[i] if len(texts) == len(names) else None)
                for i, (n, v) in enumerate(zip(names, lot_vals))]
    share = value / len(unique)
    return [row(i, n, wid(i, unique), share, currency, "split_equal", "C",
                f"{field}={value} split equally over {len(unique)} winners")
            for i, n in enumerate(unique)]


def parse_notice(raw: dict, query: str, retrieved_at: datetime, cpv_include: set[str],
                 countries3: set[str]) -> tuple[dict, list[dict]]:
    nid = raw["publication-number"]
    value, currency, field = notice_value(raw)
    verify_value(raw, field, value)
    cpv_all = list(dict.fromkeys(_list(raw.get("classification-cpv"))))
    buyer_c = [c for c in _list(raw.get("buyer-country")) if c in countries3] or _list(raw.get("buyer-country"))
    names = _first_lang(raw.get("winner-name"))
    awards = parse_awards(raw, value, currency, field)
    ids = _list(raw.get("winner-identifier"))
    tender = dict(
        notice_id=nid,
        country=ISO3_TO_ISO2.get(buyer_c[0], buyer_c[0]) if buyer_c else None,
        buyer="; ".join(dict.fromkeys(_first_lang(raw.get("buyer-name")))) or None,
        cpv=";".join(c for c in cpv_all if c in cpv_include),
        award_date=award_date(raw),
        winner_name="; ".join(dict.fromkeys(a["winner_name"] for a in awards)) or ("; ".join(names) or None),
        winner_id="; ".join(dict.fromkeys(ids)) if ids else None,
        notice_type=raw.get("notice-type"),
        publication_date=_date(raw.get("publication-date")),
        buyer_legal_type=";".join(dict.fromkeys(_list(raw.get("buyer-legal-type")))) or None,
        cpv_all=";".join(cpv_all),
        value=value,
        currency=currency,
        value_field=field,
        n_winners=len(awards),
        source_url=NOTICE_URL.format(id=nid),
        api_query=query,
        retrieved_at=retrieved_at,
        raw_json=json.dumps({k: v for k, v in raw.items() if k != "links"}, ensure_ascii=False, sort_keys=True),
        is_framework=is_framework(raw),
        source="ted",
        title=(_first_lang(raw.get("title-proc")) or [None])[0],
        _lot_texts=lot_texts(raw),
    )
    for a in awards:
        a["notice_id"] = nid
    return tender, awards


def xml_awards(notice_id: str, url: str, result: ted_xml.XmlResult) -> list[dict]:
    """Award rows from a parsed notice XML. Evidence quotes the XML element the value came from."""
    return [dict(notice_id=notice_id, seq=i, winner_name=a.winner_name, winner_id=a.winner_id, value=a.value,
                 currency=a.currency, alignment="xml", grade=a.grade, evidence=a.evidence, source_url=url,
                 scope_note=None, _own_text=result.lot_texts.get(a.lot_id or ""))
            for i, a in enumerate(result.awards)]


XML_TOLERANCE = 0.25


def xml_reconciles(res: ted_xml.XmlResult, notice_value: float | None, tol: float = XML_TOLERANCE) -> tuple[bool, str]:
    """Use XML rows only if every winner is named and the XML total matches the notice value within tol.
    Catches buyers that repeat the notice ceiling on every lot, and lots with no/zero amounts."""
    if not res.awards:
        return False, "no winning tenders in XML"
    if any(not a.winner_name for a in res.awards):
        return False, "winner name missing in XML"
    if notice_value:
        ratio = res.total() / notice_value
        if abs(ratio - 1) > tol:
            return False, f"XML total {res.total():,.0f} vs notice {notice_value:,.0f} (x{ratio:.2f})"
    return True, ""


def core_codes(cfg: dict) -> set[str]:
    return {c for c, v in cfg["include"].items() if v["tier"] == "core"}


def select_for_xml(tenders: list[dict], awards: list[dict], core: set[str], n: int = XML_TOP_N) -> list[str]:
    """Top-n notices by EUR value with any grade B/C award, in core tier, with at least one non-IVD winner."""
    by_notice: dict[str, list[dict]] = {}
    for a in awards:
        by_notice.setdefault(a["notice_id"], []).append(a)
    cands = []
    for t in tenders:
        aw = by_notice.get(t["notice_id"], [])
        if (t.get("value_eur") and set(t["cpv"].split(";")) & core
                and any(a["grade"] in ("B", "C") for a in aw)
                and any(winner_group(a["winner_name"]) != IVD for a in aw)):
            cands.append((t["value_eur"], t["notice_id"]))
    return [nid for _, nid in sorted(cands, reverse=True)[:n]]


# ---------------------------------------------------------------- orchestration

def run_plan(today: date | None = None) -> tuple[list[str], list[tuple[Slice, int]]]:
    cfg = load_cpv_config()
    cpvs = sorted(cfg["include"])
    today = today or date.today()
    slices = year_slices(cluster_countries_iso3(), lookback_start(today), today)
    client = TedClient()
    try:
        return cpvs, plan(client, cpvs, slices)
    finally:
        client.close()


def run_fetch(today: date | None = None, db_path: Path | str | None = None, xml_top_n: int = XML_TOP_N) -> dict:
    cfg = load_cpv_config()
    cpvs = sorted(cfg["include"])
    countries3 = set(cluster_countries_iso3())
    today = today or date.today()
    slices = year_slices(sorted(countries3), lookback_start(today), today)

    # 1) network: TED search
    client = TedClient()
    raws: dict[str, tuple[dict, str, datetime]] = {}
    try:
        planned = plan(client, cpvs, slices)
        for s, n in planned:
            if n == 0:
                continue
            q = build_query(cpvs, s)
            got = fetch_query(client, q, n)
            ts = datetime.now(timezone.utc).replace(tzinfo=None)
            for r in got:
                raws.setdefault(r["publication-number"], (r, q, ts))  # dedupe joint procurements
            print(f"  {s.label()}: {len(got)}/{n}")
    finally:
        client.close()

    tenders, awards = [], []
    for raw, q, ts in raws.values():
        t, a = parse_notice(raw, q, ts, set(cpvs), countries3)
        tenders.append(t)
        awards.extend(a)

    # 2) network: ECB FX for non-EUR currencies (needed to rank notices by EUR value)
    currencies = {t["currency"] for t in tenders} | {a["currency"] for a in awards}
    dates = [t["award_date"] or t["publication_date"] for t in tenders]
    fx = ecb_fx.fetch_monthly_rates(currencies, min(dates).strftime("%Y-%m")) if dates else []
    rates = {(r.currency, r.month): r.units_per_eur for r in fx}
    apply_fx(tenders, awards, rates)

    # 3) network: notice XML for the top-N grade B/C notices only
    targets = select_for_xml(tenders, awards, core_codes(cfg), xml_top_n)
    xmls = ted_xml.fetch_xml(targets)
    upgraded, xml_failed = 0, [nid for nid in targets if nid not in xmls]
    xml_rejected: dict[str, str] = {}
    for nid, (url, text) in xmls.items():
        try:
            res = ted_xml.parse_notice_xml(text)
        except Exception as e:  # malformed XML: keep search-API rows
            xml_failed.append(nid)
            print(f"  XML parse failed for {nid}: {e}")
            continue
        t = next(t for t in tenders if t["notice_id"] == nid)
        ok, why = xml_reconciles(res, t["value"])
        if not ok:
            xml_rejected[nid] = why
            continue
        awards = [a for a in awards if a["notice_id"] != nid] + xml_awards(nid, url, res)
        t["_lot_texts"] = list(res.lot_texts.values())
        upgraded += res.fully_resolved
    apply_fx(tenders, awards, rates)
    apply_scope_flags(tenders, awards)

    # 4) DB: short batches, no network from here on
    save(tenders, awards, fx, db_path)
    return {"notices": len(tenders), "awards": len(awards), "fx_rates": len(fx),
            "missing_eur": sum(t["value"] is not None and t["value_eur"] is None for t in tenders),
            "xml_targets": len(targets), "xml_fully_resolved": upgraded, "xml_failed": xml_failed,
            "xml_rejected": xml_rejected}


def print_cpv() -> None:
    cfg = load_cpv_config()
    print(f"{'CPV':<10}{'tier':<7}{'category':<31}label")
    for code, c in sorted(cfg["include"].items()):
        print(f"{code:<10}{c['tier']:<7}{c['category']:<31}{c['label']}" + (f"  [{c['note']}]" if c.get("note") else ""))
    print("\nExcluded:")
    for code, why in sorted(cfg["exclude"].items()):
        print(f"{code:<10}{why}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cpv", "plan", "fetch"])
    args = ap.parse_args(argv)
    if args.command == "cpv":
        print_cpv()
    elif args.command == "plan":
        cpvs, planned = run_plan()
        for s, n in planned:
            print(f"{s.label():<30}{n:>7}")
        print(f"{'total (before dedupe)':<30}{sum(n for _, n in planned):>7}")
    else:
        print(run_fetch())


if __name__ == "__main__":
    main()
