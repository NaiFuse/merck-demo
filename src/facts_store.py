"""Batched writes of documents / facts / assumptions (rule 7: open, write one batch, close).

    uv run python -m src.facts_store load [--dry-run]

`load` rebuilds every row from the offline caches (EDGAR cache, PDF extraction cache, assumption YAMLs) and the
peer restatement, then writes them in one short transaction. No network or LLM calls happen here.
Idempotent: a document's facts are replaced as a whole; assumptions are replaced by key.
Never touches the tenders / tender_awards tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.db import connect, init_db, table_counts
from src.extract.numbers import quote_contains_value


@dataclass
class FactRow:
    """One fact plus the document it came from."""
    source: str               # 'edgar' | 'manual_pdf' | 'derived'
    url_or_path: str          # filing URL, data/raw path, or 'derived:<model>'
    retrieved_at: datetime
    entity: str
    period: str
    metric: str
    value: float
    unit: str
    region: str | None
    segment: str | None
    quote: str                # exact quote/row (PDF facts: 'p.<page>: <quote>')
    grade: str
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AssumptionRow:
    key: str
    value_low: float | None
    value_base: float
    value_high: float | None
    source: str
    grade: str
    note: str | None = None


def check_rows(rows: list[FactRow]) -> None:
    """Rule 1 gate before any write: every non-derived fact's quote must contain its value."""
    for r in rows:
        if r.grade not in ("A", "B", "C"):
            raise ValueError(f"bad grade {r.grade!r}: {r}")
        if r.source == "edgar":
            ok = quote_contains_value(r.quote, r.value * 1e6, "USD") if r.unit == "USD_m" else \
                quote_contains_value(r.quote, r.value, r.unit)
        else:
            ok = quote_contains_value(r.quote, r.value, r.unit)
        if not ok:
            raise ValueError(f"quote does not contain value {r.value} {r.unit}: {r.quote[:200]}")


def save(facts: list[FactRow], assumptions: list[AssumptionRow] = (), db_path: Path | str | None = None) -> dict:
    """Write one batch. Returns counts. All inputs must be fully built before this is called (rule 7)."""
    check_rows(facts)
    docs: dict[tuple[str, str], datetime] = {}
    for r in facts:
        k = (r.source, r.url_or_path)
        docs[k] = max(docs.get(k, r.retrieved_at), r.retrieved_at)
    with connect(db_path) as con:
        init_db(con)
        con.execute("BEGIN")
        try:
            doc_ids = {}
            for (source, url), ts in docs.items():
                row = con.execute("SELECT id FROM documents WHERE source = ? AND url_or_path = ?", [source, url]).fetchone()
                if row:
                    con.execute("UPDATE documents SET retrieved_at = ? WHERE id = ?", [ts, row[0]])
                    con.execute("DELETE FROM facts WHERE document_id = ?", [row[0]])
                    doc_ids[(source, url)] = row[0]
                else:
                    doc_ids[(source, url)] = con.execute(
                        "INSERT INTO documents (source, url_or_path, retrieved_at) VALUES (?, ?, ?) RETURNING id",
                        [source, url, ts]).fetchone()[0]
            con.executemany(
                "INSERT INTO facts (document_id, entity, period, metric, value, unit, region, segment, quote, grade) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [[doc_ids[(r.source, r.url_or_path)], r.entity, r.period, r.metric, r.value, r.unit, r.region,
                  r.segment, r.quote, r.grade] for r in facts])
            if assumptions:
                con.executemany("DELETE FROM assumptions WHERE key = ?", [[a.key] for a in assumptions])
                con.executemany(
                    "INSERT INTO assumptions (key, value_low, value_base, value_high, source, grade, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [[a.key, a.value_low, a.value_base, a.value_high, a.source, a.grade, a.note] for a in assumptions])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return {"documents": len(docs), "facts": len(facts), "assumptions": len(assumptions),
                "table_counts": table_counts(con)}


def build_all() -> tuple[list[FactRow], list[AssumptionRow]]:
    """Every row from caches + config (offline)."""
    from src.connectors import edgar
    from src.extract import pdf
    from src.models import peer_restatement

    facts, problems = edgar.load_all(online=False)
    if problems:
        raise RuntimeError("EDGAR cross-check failed: " + "; ".join(problems))
    accepted, _ = pdf.verify_all_cached()
    facts += accepted
    result = peer_restatement.run_all(facts)
    facts += result.fact_rows()
    return facts, result.assumption_rows()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load")
    ld.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    facts, assumptions = build_all()   # all reads/verification before the connection opens
    print(f"built {len(facts)} facts, {len(assumptions)} assumptions")
    if args.dry_run:
        check_rows(facts)
        print("dry run: rule-1 checks passed, nothing written")
        return
    out = save(facts, assumptions)
    for t, n in out["table_counts"].items():
        print(f"{t:<14}{n:>8}")


if __name__ == "__main__":
    main()
