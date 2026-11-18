"""Extract disclosed figures from the PDFs in data/raw with Claude, each with an exact quote and page, then verify.

    uv run python -m src.extract.pdf run [paths...] [--refresh]   # calls Claude for uncached PDFs
    uv run python -m src.extract.pdf report                       # offline: re-verify cached responses

Rule 2: the LLM only reads the PDF and copies figures + quotes; it never calculates. Code then checks every item:
  1. the quote appears in the local pypdf text of the stated page (normalised: NFKC, whitespace, hyphenation);
     if it appears on exactly one other page, the page is corrected and the item says so;
  2. the quote contains the value (src/extract/numbers.py).
Items failing a check are rejected (kept in the report, never stored). Responses are cached per file SHA-256 in
data/raw/extract_cache/ so reruns are free and the raw model output stays auditable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from src.extract.numbers import quote_contains_value, unit_multiplier
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT, get_settings

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DIRS = [RAW_DIR / "peers", RAW_DIR / "registers"]
CACHE_DIR = RAW_DIR / "extract_cache"
PROMPT_VERSION = "m3-v1"
UNITS = ["USD_m", "USD_bn", "EUR", "EUR_k", "EUR_m", "EUR_bn", "%", "count"]

# company slug in the file name -> entity key in the facts table
ENTITY = {
    "thermo-fisher": "thermo_fisher",
    "avantor": "avantor",
    "fisher-scientific-gmbh": "fisher_scientific_gmbh",
    "vwr-international-gmbh": "vwr_international_gmbh",
    "carl-roth-gmbh-co-kg": "carl_roth",
    "merck-kgaa": "merck_kgaa",
}

TARGETS = {
    "earnings-release": (
        "Quarterly and year-to-date revenue (total, by segment, by region if given); organic revenue growth (total, "
        "by segment, by region, by end market such as academic & government); full-year revenue guidance. "
        "metric names: revenue, organic_growth, reported_growth, revenue_guidance_low, revenue_guidance_high."),
    "jahresabschluss": (
        "Umsatzerlöse (total and every split given: by region/Inland/Ausland/EU, by product group, by customer group), "
        "number of employees, fiscal year. Use the current-year column unless the prior year is clearly labelled. "
        "metric names: revenue, revenue_share, employees. Units for amounts in the table header (e.g. TEUR -> EUR_k)."),
    "aufsichtsratsbericht": "Any revenue (Umsatz) or growth figures. metric names: revenue, revenue_growth.",
    "aufsichtsratsprotokoll": "Any revenue (Umsatz), budget or growth figures. metric names: revenue, revenue_budget, revenue_growth.",
}

SYSTEM = """You extract disclosed financial figures from a company document for an audit trail.
Rules:
- Copy figures exactly as printed. Never calculate, convert, sum or infer a number that is not printed.
- For every figure give the 1-based PDF page number and an exact, verbatim quote (one sentence or one table row,
  max ~300 characters) from that page that contains the number as printed. Do not paraphrase or fix typos.
- value is the number as printed in the quote, in the given unit (e.g. "$10.9 billion" -> value 10.9, unit USD_bn;
  a table row in millions -> unit USD_m; "TEUR 1.234" -> value 1234, unit EUR_k; "3%" -> value 3, unit %).
- period: 'FY2025' for a fiscal year, '2026Q3' for a quarter, 'YTD2026Q3' for year-to-date.
- region / segment: short English labels as named in the document (e.g. 'Europe', 'Germany', 'Life Sciences Solutions',
  'academic and government'), or null.
- Only extract what the target list asks for. If nothing matches, return an empty list."""

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "period": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string", "enum": UNITS},
                    "region": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "segment": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "page": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["metric", "period", "value", "unit", "region", "segment", "page", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- file names and page text

@dataclass(frozen=True)
class DocInfo:
    path: Path
    country: str
    company: str
    fiscal_year: str
    doc_type: str

    @property
    def entity(self) -> str:
        return ENTITY.get(self.company, self.company.replace("-", "_"))

    @property
    def rel_path(self) -> str:
        try:
            return str(self.path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.path)


def parse_filename(path: Path) -> DocInfo:
    """<country>_<company>_<fiscal-year>_<document>.pdf (CLAUDE.md naming rule)."""
    parts = path.stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"file name does not follow <country>_<company>_<fiscal-year>_<document>.pdf: {path.name}")
    return DocInfo(path, *parts)


def page_texts(path: Path) -> list[str]:
    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_DASHES).replace("\u00ad", "")
    # control characters: pypdf emits e.g. \x00 for unmapped ligature glyphs ('re\x00ecting' for 'reflecting')
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)       # hyphenated line breaks
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", "", t).lower()               # PDF table text has arbitrary spacing


def find_quote_pages(quote: str, pages: list[str]) -> list[int]:
    """1-based pages whose normalised text contains the normalised quote."""
    q = normalise(quote)
    if len(q) < 8:
        return []
    return [i + 1 for i, t in enumerate(pages) if q in normalise(t)]


# ---------------------------------------------------------------- verification

@dataclass
class Checked:
    item: dict
    ok: bool
    reason: str
    page: int | None = None


def verify_item(item: dict, pages: list[str]) -> Checked:
    if item.get("unit") not in UNITS:
        return Checked(item, False, f"unknown unit {item.get('unit')!r}")
    if not quote_contains_value(item["quote"], item["value"], item["unit"]):
        return Checked(item, False, "quote does not contain the value")
    stated = item.get("page")
    found = find_quote_pages(item["quote"], pages)
    if stated in found:
        return Checked(item, True, "ok", stated)
    if len(found) == 1:
        return Checked(item, True, f"page corrected {stated} -> {found[0]}", found[0])
    if not found:
        return Checked(item, False, f"quote not found on page {stated} or any other page")
    return Checked(item, False, f"quote not on stated page {stated}; found on several pages {found}")


def to_fact_row(doc: DocInfo, c: Checked, retrieved_at: datetime) -> FactRow:
    it = c.item
    return FactRow("manual_pdf", doc.rel_path, retrieved_at, doc.entity, it["period"], it["metric"], float(it["value"]),
                   it["unit"], it.get("region"), it.get("segment"), it["quote"], "A",
                   {"page": c.page, "check": c.reason})


# ---------------------------------------------------------------- LLM call + cache

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_file(path: Path, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{sha256(path)}.json"


def build_request(doc: DocInfo, model: str) -> dict:
    data = base64.standard_b64encode(doc.path.read_bytes()).decode("ascii")
    targets = TARGETS.get(doc.doc_type, "Revenue and revenue growth figures. metric names: revenue, organic_growth.")
    return {
        "model": model,
        "max_tokens": 32000,
        "system": SYSTEM,
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
            {"type": "text", "text": f"Document: {doc.path.name} (company: {doc.company}, fiscal year {doc.fiscal_year}, "
                                     f"type: {doc.doc_type}).\nTargets: {targets}"},
        ]}],
    }


def call_claude(client, request: dict) -> tuple[str, dict]:
    """Returns (JSON text, usage). Streams, because long PDFs can take a while."""
    with client.messages.stream(**request) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {getattr(msg, 'stop_details', None)}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("output hit max_tokens; split the document")
    text = next(b.text for b in msg.content if b.type == "text")
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return text, usage


def extract(path: Path, client=None, refresh: bool = False, cache_dir: Path = CACHE_DIR) -> dict:
    """Cached raw extraction for one PDF. Calls Claude only when there is no cache (or refresh)."""
    doc = parse_filename(path)
    cf = cache_file(path, cache_dir)
    if cf.exists() and not refresh:
        return json.loads(cf.read_text(encoding="utf-8"))
    model = get_settings().anthropic_model
    if not model:
        raise RuntimeError("ANTHROPIC_MODEL is not set in .env")
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())
    text, usage = call_claude(client, build_request(doc, model))
    record = {"file": doc.rel_path, "sha256": sha256(path), "model": model, "prompt_version": PROMPT_VERSION,
              "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "usage": usage,
              "response_text": text}
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def verify_record(path: Path, record: dict) -> tuple[list[FactRow], list[Checked]]:
    doc = parse_filename(path)
    pages = page_texts(path)
    items = json.loads(record["response_text"])["facts"]
    retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
    checked = [verify_item(it, pages) for it in items]
    return [to_fact_row(doc, c, retrieved_at) for c in checked if c.ok], [c for c in checked if not c.ok]


def dedupe_facts(rows: list[FactRow]) -> tuple[list[FactRow], list[FactRow]]:
    """Same figure stated twice in one document (e.g. 'Umsatz 101.035 TEUR' in the management report and
    'Umsatzerlöse 101.034.745,26' in the income statement): same entity/period/metric/region/segment, values equal
    within the coarser figure's display rounding. Keeps the more precise one. Returns (kept, dropped)."""
    def base(r):
        return r.value * unit_multiplier(r.unit) if r.unit != "%" else r.value

    def step(r):  # half a display unit of the coarser figure, in base units
        if r.unit == "%":
            return 0.05
        return 0.5 * unit_multiplier(r.unit) * (1 if float(r.value).is_integer() else 0.1)

    kept: list[FactRow] = []
    dropped: list[FactRow] = []
    for r in sorted(rows, key=lambda x: (unit_multiplier(x.unit) if x.unit != "%" else 0, x.extra.get("page") or 0)):
        dup = next((k for k in kept if (k.url_or_path, k.entity, k.period, k.metric, k.region, k.segment)
                    == (r.url_or_path, r.entity, r.period, r.metric, r.region, r.segment)
                    and (k.unit == "%") == (r.unit == "%")
                    and abs(base(k) - base(r)) <= max(step(k), step(r))), None)
        if dup is None:
            kept.append(r)
        else:
            dup.extra.setdefault("duplicates", []).append(f"p.{r.extra.get('page')}: {r.value} {r.unit}")
            dropped.append(r)
    return kept, dropped


def pdf_paths(paths: list[Path] | None = None) -> list[Path]:
    out = []
    for p in paths or DEFAULT_DIRS:
        out += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return out


def verify_all_cached(paths: list[Path] | None = None, cache_dir: Path = CACHE_DIR) -> tuple[list[FactRow], list[tuple[str, Checked]]]:
    """Offline: re-verify every cached response (PDFs without a cache are skipped)."""
    accepted, rejected = [], []
    for p in pdf_paths(paths):
        cf = cache_file(p, cache_dir)
        if not cf.exists():
            continue
        ok, bad = verify_record(p, json.loads(cf.read_text(encoding="utf-8")))
        ok, _ = dedupe_facts(ok)
        accepted += ok
        rejected += [(p.name, c) for c in bad]
    return accepted, rejected


def print_report(accepted: list[FactRow], rejected: list[tuple[str, Checked]], sample: int = 25) -> None:
    files = sorted({r.url_or_path for r in accepted})
    print(f"verified: {len(accepted)} facts from {len(files)} files; rejected: {len(rejected)}")
    for r in accepted[:sample]:
        print(f"  {r.entity:<24}{r.period:<11}{r.metric:<22}{(r.region or r.segment or '-')[:24]:<26}"
              f"{r.value:>12,.2f} {r.unit:<6} p.{r.extra['page']}  \"{r.quote[:70]}\"")
    if rejected:
        print("rejected:")
        for name, c in rejected:
            print(f"  {name}: {c.reason} | {c.item.get('metric')} {c.item.get('value')} {c.item.get('unit')} "
                  f"p.{c.item.get('page')} \"{c.item.get('quote', '')[:70]}\"")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("paths", nargs="*", type=Path)
    r.add_argument("--refresh", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("paths", nargs="*", type=Path)
    args = ap.parse_args(argv)
    paths = pdf_paths(args.paths or None)
    if args.cmd == "run":
        for p in paths:
            rec = extract(p, refresh=args.refresh)
            print(f"{p.name}: {len(json.loads(rec['response_text'])['facts'])} items, usage {rec['usage']}")
    print_report(*verify_all_cached(paths))


if __name__ == "__main__":
    main()
