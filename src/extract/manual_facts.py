"""Facts read from a rendered page image (config/manual_facts.yaml), verified against the PDF text layer.

    uv run python -m src.extract.manual_facts

Two kinds of entries:
  - table facts: the text layer scrambles a table, so the label/value pairing is read from the rendered page; every
    quote fragment must be verbatim on the page, the value fragment inside one of them, the label on the page;
  - plain quotes (no label / value_fragment): one verbatim sentence on the page that contains the value.
Facts that fail are rejected with the reason (never stored).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.extract.numbers import quote_contains_value
from src.facts_store import FactRow
from src.settings import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "config" / "manual_facts.yaml"


def verify(item: dict, pages: list[str]) -> list[str]:
    from src.extract.pdf import find_quote_pages, normalise

    page = item["page"]
    label = [item["label"]] if item.get("label") else []
    problems = [f"fragment not on p.{page}: {q!r}" for q in item["quotes"] + label
                if page not in find_quote_pages(q, pages)]
    frag = item.get("value_fragment") or item["quotes"][0]
    if not any(normalise(frag) in normalise(q) for q in item["quotes"]):
        problems.append(f"value fragment {frag!r} not inside any quote")
    if not quote_contains_value(frag, item["value"], item["unit"]):
        problems.append(f"value fragment {frag!r} does not contain {item['value']} {item['unit']}")
    return problems


def quote_text(item: dict) -> str:
    """Stored facts.quote: page, verbatim fragments, and how the label/value pairing was read."""
    if not item.get("label"):
        return f"p.{item['page']}: {item['quotes'][0]}" + (f" [{item['note']}]" if item.get("note") else "")
    frags = " | ".join(f'"{q}"' for q in item["quotes"])
    return (f"p.{item['page']}: row {item['label']!r} = {item['value_fragment']!r}; verbatim text-layer runs: {frags}. "
            f"Pairing: {item['pairing']}")


def load_all(path: Path = CONFIG, root: Path = PROJECT_ROOT) -> tuple[list[FactRow], list[str]]:
    from src.extract.pdf import page_texts

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows, problems, cache = [], [], {}
    for item in cfg.get("facts", []):
        f = root / item["file"]
        if not f.exists():
            problems.append(f"{item['file']}: missing")
            continue
        pages = cache.setdefault(f, page_texts(f))
        p = verify(item, pages)
        if p:
            problems += [f"{item['entity']} {item['metric']}: {x}" for x in p]
            continue
        retrieved = datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).replace(tzinfo=None)
        rows.append(FactRow("manual_pdf", item["file"], retrieved, item["entity"], item["period"], item["metric"],
                            float(item["value"]), item["unit"], item.get("region"), item.get("segment"),
                            quote_text(item), item["grade"]))
    return rows, problems


if __name__ == "__main__":
    rows, problems = load_all()
    for r in rows:
        print(f"OK  {r.entity} {r.period} {r.metric} = {r.value:g} {r.unit} [{r.grade}] {r.url_or_path}")
    for p in problems:
        print("REJECTED", p)
