"""UK Find a Tender Service (FTS) award notices via the official OCDS API (no key).

    uv run python -m src.connectors.uk_fts plan             # page through one sample day, estimate runtime
    uv run python -m src.connectors.uk_fts fetch [--refresh] # download, convert FX, write to DuckDB
    uv run python -m src.connectors.uk_fts load-cache        # no FTS calls: load only the cached months

API: GET https://www.find-tender.service.gov.uk/api/1.0/ocdsReleasePackages
     params stages=award, updatedFrom/updatedTo (YYYY-MM-DDTHH:MM:SS), limit<=100; next page = links.next.
FTS has no server-side CPV filter, so every award release in the window is paged and filtered locally.
Only matching releases are kept: cached per month in data/raw/uk_fts/<YYYY-MM>.jsonl so an interrupted run
resumes. FTS went live 1 Jan 2021 (UK left TED); below-threshold UK tenders are on Contracts Finder instead.

Rule 7: all network calls (FTS pages, ECB FX) finish before tender_store.save() opens DB connections.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from src.connectors import ecb_fx
from src.connectors.ted import load_cpv_config, lookback_start
from src.connectors.tender_store import apply_fx, apply_scope_flags, save
from src.settings import PROJECT_ROOT

API_URL = "https://www.find-tender.service.gov.uk/api/1.0/ocdsReleasePackages"
NOTICE_URL = "https://www.find-tender.service.gov.uk/Notice/{id}"
NOTICE_PREFIX = "FTS-"          # keeps FTS ids (nnnnnn-yyyy) apart from TED ids
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "uk_fts"
PAGE_LIMIT = 100
MIN_INTERVAL_S = 1.0


# ---------------------------------------------------------------- HTTP

class FtsClient:
    def __init__(self, client: httpx.Client | None = None, min_interval: float = MIN_INTERVAL_S, retries: int = 6,
                 sleep=time.sleep):
        self._client = client or httpx.Client(timeout=120, headers={"User-Agent": "merck-ds-demo/0.1 (research)"})
        self._min_interval = min_interval
        self._retries = retries
        self._sleep = sleep
        self._last = 0.0

    def close(self) -> None:
        self._client.close()

    def get(self, url: str, params: dict | None = None) -> tuple[str, dict]:
        """Returns (final URL, JSON). Honours Retry-After on 429/503 as the API docs require."""
        delay = 2.0
        for attempt in range(self._retries):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError:
                if attempt == self._retries - 1:
                    raise
                self._sleep(delay)
                delay *= 2
                continue
            if resp.status_code in (429, 503) or resp.status_code >= 500:
                if attempt == self._retries - 1:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
                delay *= 2
                continue
            if resp.status_code >= 400:
                raise ValueError(f"FTS API {resp.status_code}: {resp.text[:500]}")
            return str(resp.url), resp.json()
        raise AssertionError("unreachable")


def iter_pages(client: FtsClient, start: datetime, end: datetime):
    """Yield (page URL, release package) for all award releases updated in [start, end)."""
    params = {"stages": "award", "limit": PAGE_LIMIT,
              "updatedFrom": start.strftime("%Y-%m-%dT%H:%M:%S"), "updatedTo": end.strftime("%Y-%m-%dT%H:%M:%S")}
    url, pkg = client.get(API_URL, params)
    while True:
        yield url, pkg
        nxt = (pkg.get("links") or {}).get("next")
        if not nxt or not pkg.get("releases"):
            return
        url, pkg = client.get(nxt)


def month_windows(start: date, end: date) -> list[tuple[str, datetime, datetime]]:
    """[(label 'YYYY-MM', from, to)] covering start..end (end inclusive), clipped to the window."""
    out, cur = [], date(start.year, start.month, 1)
    stop = datetime.combine(end + timedelta(days=1), datetime.min.time())
    while cur <= end:
        nxt = date(cur.year + cur.month // 12, cur.month % 12 + 1, 1)
        a = datetime.combine(max(cur, start), datetime.min.time())
        b = min(datetime.combine(nxt, datetime.min.time()), stop)
        out.append((f"{cur:%Y-%m}", a, b))
        cur = nxt
    return out


# ---------------------------------------------------------------- parsing (pure functions)

def _date(s: str | None) -> date | None:
    return date.fromisoformat(s[:10]) if s else None


def release_cpvs(r: dict) -> list[str]:
    """All CPV codes on the release (main + items + additional), 8 digits without check digit."""
    t = r.get("tender") or {}
    cls = [t.get("classification")] + [i.get("classification") for i in t.get("items") or []]
    for i in t.get("items") or []:
        cls += i.get("additionalClassifications") or []
    codes = [str(c["id"])[:8] for c in cls if c and c.get("scheme") == "CPV" and c.get("id")]
    return list(dict.fromkeys(codes))


def keep_release(r: dict, cpv_include: set[str], start: date, end: date) -> bool:
    d = _date(r.get("date"))
    return bool(r.get("awards")) and d is not None and start <= d <= end and bool(set(release_cpvs(r)) & cpv_include)


def award_values(r: dict) -> dict[str, tuple[float, str, str]]:
    """{award id: (amount, currency, evidence)}. award.value if given, else the sum of contracts[].value
    linked by awardID (FTS usually puts the value on the contract)."""
    out: dict[str, tuple[float, str, str]] = {}
    for i, a in enumerate(r.get("awards") or []):
        v = a.get("value") or {}
        if v.get("amount") is not None:
            out[a["id"]] = (float(v["amount"]), v.get("currency"), f"awards[{i}].value.amount={v['amount']} {v.get('currency')}")
    for j, c in enumerate(r.get("contracts") or []):
        v = c.get("value") or {}
        aid = c.get("awardID")
        if aid and aid not in out and v.get("amount") is not None:
            out[aid] = (float(v["amount"]), v.get("currency"), f"contracts[{j}].value.amount={v['amount']} {v.get('currency')}")
        elif aid and v.get("amount") is not None and out[aid][2].startswith("contracts"):
            amt, cur, ev = out[aid]
            out[aid] = (amt + float(v["amount"]), cur, f"{ev} + contracts[{j}].value.amount={v['amount']}")
    return out


def dedupe(releases: list[dict]) -> list[tuple[dict, set[str]]]:
    """One procedure (ocid) can publish several award releases (F03 then awardUpdate; UK6 then UK7).
    Keep, per ocid and lot, only the awards of the latest release. Returns [(release, kept award ids)]."""
    by_id: dict[str, dict] = {}
    for r in releases:  # same notice seen in two update windows: keep the later version
        if r["id"] not in by_id or (r.get("date") or "") >= (by_id[r["id"]].get("date") or ""):
            by_id[r["id"]] = r
    by_ocid: dict[str, list[dict]] = {}
    for r in by_id.values():
        by_ocid.setdefault(r.get("ocid") or r["id"], []).append(r)
    out = []
    for rs in by_ocid.values():
        seen_lots: set[str] = set()
        for r in sorted(rs, key=lambda x: x.get("date") or "", reverse=True):
            kept = set()
            for a in r.get("awards") or []:
                lots = set(a.get("relatedLots") or [f"award:{a['id']}"])
                if not lots <= seen_lots:
                    kept.add(a["id"])
            for a in r.get("awards") or []:
                if a["id"] in kept:
                    seen_lots |= set(a.get("relatedLots") or [f"award:{a['id']}"])
            if kept:
                out.append((r, kept))
    return out


def _party_ids(r: dict) -> dict[str, str]:
    out = {}
    for p in r.get("parties") or []:
        ident = p.get("identifier") or {}
        if p.get("id") and ident.get("scheme") and ident.get("id"):
            out[p["id"]] = f"{ident['scheme']}:{ident['id']}"
    return out


def parse_release(r: dict, kept: set[str], api_url: str, retrieved_at: datetime,
                  cpv_include: set[str]) -> tuple[dict, list[dict]]:
    rid = r["id"]
    nid = NOTICE_PREFIX + rid
    t = r.get("tender") or {}
    values = award_values(r)
    pids = _party_ids(r)
    lots = {str(l.get("id")): " | ".join(x for x in (l.get("title"), l.get("description")) if x)
            for l in t.get("lots") or []}
    signed = {c.get("awardID"): _date(c.get("dateSigned")) for c in r.get("contracts") or [] if c.get("dateSigned")}

    awards, seq, total, cur, award_dates = [], 0, 0.0, None, []
    for a in r.get("awards") or []:
        if a["id"] not in kept or a.get("status") not in (None, "active"):
            continue
        sups = a.get("suppliers") or []
        rel = [str(x) for x in a.get("relatedLots") or []]
        own_text = " || ".join(lots[x] for x in rel if lots.get(x)) or None
        d = _date(a.get("date")) or signed.get(a["id"])
        if d:
            award_dates.append(d)
        val = values.get(a["id"])
        if val:
            total += val[0]
            cur = cur or val[1]
        if not sups:
            continue
        if val is None:
            rows = [(s, None, "no_value", "C", "no award/contract value") for s in sups]
        elif len(sups) == 1:
            rows = [(sups[0], val[0], "lot" if rel else "single", "A", val[2])]
        else:
            rows = [(s, val[0] / len(sups), "split_equal", "C", f"{val[2]} split equally over {len(sups)} suppliers")
                    for s in sups]
        for s, v, alignment, grade, ev in rows:
            awards.append(dict(notice_id=nid, seq=seq, winner_name=s.get("name"),
                               winner_id=pids.get(s.get("id")) or s.get("id"), value=v,
                               currency=val[1] if val else None, alignment=alignment, grade=grade,
                               evidence=f"{a['id']}: {ev}", source_url=None, scope_note=None, _own_text=own_text))
            seq += 1

    cpv_all = release_cpvs(r)
    buyer = (r.get("buyer") or {}).get("name") or next(
        (p.get("name") for p in r.get("parties") or [] if "buyer" in (p.get("roles") or [])), None)
    buyer_party = next((p for p in r.get("parties") or [] if "buyer" in (p.get("roles") or [])), {})
    country = (buyer_party.get("address") or {}).get("countryName")
    classifications = [c.get("id") or c.get("description") for c in
                       (buyer_party.get("details") or {}).get("classifications") or []]
    has_value = any(a["id"] in values for a in r.get("awards") or [] if a["id"] in kept)
    tender = dict(
        notice_id=nid,
        country="GB" if country in (None, "United Kingdom", "UK", "GB") else country,
        buyer=buyer,
        cpv=";".join(c for c in cpv_all if c in cpv_include),
        award_date=min(award_dates) if award_dates else _date(r.get("date")),
        winner_name="; ".join(dict.fromkeys(a["winner_name"] for a in awards if a["winner_name"])) or None,
        winner_id="; ".join(dict.fromkeys(a["winner_id"] for a in awards if a["winner_id"])) or None,
        notice_type="fts:" + "+".join(r.get("tag") or []),
        publication_date=_date(r.get("date")),
        buyer_legal_type=";".join(str(c) for c in classifications if c) or None,
        cpv_all=";".join(cpv_all),
        value=total if has_value else None,
        currency=cur,
        value_field="awards/contracts value.amount (sum of kept awards)" if has_value else None,
        n_winners=len(awards),
        source_url=NOTICE_URL.format(id=rid),
        api_query=api_url,
        retrieved_at=retrieved_at,
        raw_json=json.dumps(r, ensure_ascii=False, sort_keys=True),
        is_framework=bool((t.get("techniques") or {}).get("hasFrameworkAgreement")),
        title=t.get("title"),
        source="uk_fts",
        _lot_texts=[x for x in lots.values() if x],
    )
    verify_total(r, kept, tender["value"])
    return tender, awards


def verify_total(r: dict, kept: set[str], value: float | None) -> None:
    """Rule 1: the stored notice value must equal the raw amounts of the kept awards."""
    if value is None:
        return
    raw = sum(v[0] for aid, v in award_values(r).items() if aid in kept)
    if abs(raw - value) > 0.005:
        raise ValueError(f"{r['id']}: value {value} does not match raw award amounts {raw}")


# ---------------------------------------------------------------- orchestration

def _cache_path(label: str, cache_dir: Path) -> Path:
    return cache_dir / f"{label}.jsonl"


MANIFEST = "_manifest.json"


def read_manifest(cache_dir: Path = CACHE_DIR) -> dict:
    p = cache_dir / MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _record_month(cache_dir: Path, label: str, matching: int, scanned: int) -> None:
    m = read_manifest(cache_dir)
    m[label] = {"matching": matching, "scanned": scanned,
                "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")}
    (cache_dir / MANIFEST).write_text(json.dumps(dict(sorted(m.items())), indent=1), encoding="utf-8")


def cached_months(cache_dir: Path = CACHE_DIR) -> list[str]:
    """'YYYY-MM' labels with a completed month file (a file exists only once the month was fully paged)."""
    return sorted(p.stem for p in cache_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].jsonl"))


def coverage(cache_dir: Path = CACHE_DIR, low_scan_ratio: float = 0.5) -> dict | None:
    """UK FTS coverage from the month cache: first/last month, gaps, and months whose scanned volume is far
    below the median (possible incomplete paging). Publication-date window = first day .. last day of month."""
    months = cached_months(cache_dir)
    if not months:
        return None
    first, last = date.fromisoformat(months[0] + "-01"), date.fromisoformat(months[-1] + "-01")
    expected = [lbl for lbl, _, _ in month_windows(first, last)]
    nxt = date(last.year + last.month // 12, last.month % 12 + 1, 1)
    man = read_manifest(cache_dir)
    scans = sorted(v["scanned"] for k, v in man.items() if k in months and v.get("scanned"))
    median = scans[len(scans) // 2] if scans else None
    low = [k for k in months if median and man.get(k, {}).get("scanned", median) < low_scan_ratio * median]
    return {"first_month": months[0], "last_month": months[-1], "start": first, "end": nxt - timedelta(days=1),
            "missing_months": [m for m in expected if m not in months], "low_scan_months": low,
            "median_scanned": median}


def fetch_month(client: FtsClient, label: str, a: datetime, b: datetime, cpv_include: set[str],
                start: date, end: date) -> tuple[list[dict], int]:
    """Network only. Returns (matching cache rows, releases scanned)."""
    rows, scanned = [], 0
    for url, pkg in iter_pages(client, a, b):
        ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        for r in pkg.get("releases") or []:
            scanned += 1
            if keep_release(r, cpv_include, start, end):
                rows.append({"api_url": url, "retrieved_at": ts, "release": r})
    return rows, scanned


def _read_cache(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def run_fetch(today: date | None = None, db_path: Path | str | None = None, refresh: bool = False,
              cache_dir: Path = CACHE_DIR, client: FtsClient | None = None, cache_only: bool = False) -> dict:
    """cache_only=True: no FTS calls at all; loads the completed month files that exist and reports the
    resulting coverage window (ECB FX is still fetched before the DB write)."""
    cfg = load_cpv_config()
    cpv_include = set(cfg["include"])
    today = today or date.today()
    start = lookback_start(today)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) network: FTS, month by month; completed months are cached (current month always refetched)
    own = client is None and not cache_only
    client = client or (None if cache_only else FtsClient())
    cached_rows: list[dict] = []
    scanned_total = 0
    try:
        for label, a, b in month_windows(start, today):
            path = _cache_path(label, cache_dir)
            current = label == f"{today:%Y-%m}"
            if cache_only:
                if path.exists():
                    rows = _read_cache(path)
                    print(f"  {label}: {len(rows)} matching (cached)")
                else:
                    print(f"  {label}: not cached – not covered")
                    rows = []
            elif path.exists() and not refresh and not current:
                rows = _read_cache(path)
                print(f"  {label}: {len(rows)} matching (cached)")
            else:
                rows, scanned = fetch_month(client, label, a, b, cpv_include, start, today)
                scanned_total += scanned
                tmp = path.with_suffix(".part")
                tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
                tmp.replace(path)
                _record_month(cache_dir, label, len(rows), scanned)
                print(f"  {label}: {len(rows)} matching of {scanned} scanned")
            cached_rows.extend(rows)
    finally:
        if own:
            client.close()

    meta = {r["release"]["id"]: r for r in cached_rows}  # latest occurrence wins
    tenders, awards = [], []
    for rel, kept in dedupe([r["release"] for r in cached_rows]):
        m = meta[rel["id"]]
        t, a = parse_release(rel, kept, m["api_url"], datetime.fromisoformat(m["retrieved_at"]), cpv_include)
        tenders.append(t)
        awards.extend(a)

    # 2) network: ECB FX (GBP and anything else)
    currencies = {t["currency"] for t in tenders} | {a["currency"] for a in awards}
    dates = [t["award_date"] or t["publication_date"] for t in tenders]
    fx = ecb_fx.fetch_monthly_rates(currencies, min(dates).strftime("%Y-%m")) if dates else []
    apply_fx(tenders, awards, {(r.currency, r.month): r.units_per_eur for r in fx})
    apply_scope_flags(tenders, awards)

    # 3) DB: short batches, no network from here on
    save(tenders, awards, fx, db_path)
    return {"releases_matched": len(cached_rows), "notices": len(tenders), "awards": len(awards),
            "scanned_this_run": scanned_total, "fx_rates": len(fx),
            "missing_eur": sum(t["value"] is not None and t["value_eur"] is None for t in tenders),
            "coverage": coverage(cache_dir)}


def run_plan(day: date | None = None) -> dict:
    """Page through one day to estimate volume and runtime of a full 36-month fetch."""
    day = day or date.today() - timedelta(days=30)
    a = datetime.combine(day, datetime.min.time())
    client = FtsClient()
    t0, pages, releases = time.monotonic(), 0, 0
    try:
        for _, pkg in iter_pages(client, a, a + timedelta(days=1)):
            pages += 1
            releases += len(pkg.get("releases") or [])
    finally:
        client.close()
    secs = time.monotonic() - t0
    days = (date.today() - lookback_start(date.today())).days
    return {"sample_day": str(day), "pages": pages, "releases": releases, "seconds": round(secs, 1),
            "est_pages_36m": pages * days, "est_minutes_36m": round(secs * days / 60)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "fetch", "load-cache"])
    ap.add_argument("--refresh", action="store_true", help="ignore the month cache and re-download")
    args = ap.parse_args(argv)
    if args.command == "plan":
        print(run_plan())
    else:
        print(run_fetch(refresh=args.refresh, cache_only=args.command == "load-cache"))


if __name__ == "__main__":
    main()
