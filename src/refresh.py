"""Full refresh from scratch into a NEW database file, then plausibility checks and a diff against the current DB.

    uv run python -m src.refresh [--out data/refresh/demo_refresh.duckdb] [--swap]

Steps (no LLM calls; network only where a loader needs it, noted below):
  1. empty DB with the full schema
  2. TED: the connector has no file cache – its raw API rows live only in the DB (tenders.raw_json). They are copied
     from the current DB and every stored value is re-verified against its raw row (rule 1); awards and FX rates too
  3. UK FTS: load-cache (cached month files; ECB FX is fetched once by the loader – network)
  4. facts / assumptions / market cells: facts_store.build_all() from all offline caches (EDGAR, PDF extraction cache,
     manual facts, EUR-Lex, Eurostat, OECD, OpenAlex, CORDIS), peer restatement, market model, reconciliation
  5. scenario + findings + PDF report (data/reports/academia_report.pdf) from the new DB
  6. plausibility checks and a table-by-table diff against the current DB
--swap replaces data/demo.duckdb with the new file (the old one is kept as data/demo.duckdb.bak) when no check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "refresh" / "demo_refresh.duckdb"


def copy_ted(src: Path, dst: Path) -> dict:
    import duckdb

    from src.connectors.ted import verify_value

    with duckdb.connect(str(dst)) as con:
        con.execute(f"ATTACH '{src}' AS old (READ_ONLY)")
        cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE "
                                          "table_catalog = 'old' AND table_name = 'tenders' ORDER BY ordinal_position")
                .fetchall()]
        acols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE "
                                           "table_catalog = 'old' AND table_name = 'tender_awards' "
                                           "ORDER BY ordinal_position").fetchall()]
        con.execute(f"INSERT INTO tenders ({', '.join(cols)}) SELECT {', '.join(cols)} FROM old.tenders "
                    "WHERE coalesce(source, 'ted') = 'ted'")
        con.execute(f"INSERT INTO tender_awards ({', '.join(acols)}) SELECT {', '.join(acols)} FROM old.tender_awards "
                    "WHERE notice_id IN (SELECT notice_id FROM tenders)")
        con.execute("INSERT OR REPLACE INTO fx_rates SELECT * FROM old.fx_rates")
        rows = con.execute("SELECT notice_id, raw_json, value_field, value FROM tenders").fetchall()
        n_aw = con.execute("SELECT count(*) FROM tender_awards").fetchone()[0]
        con.execute("DETACH old")
    bad = []
    for nid, raw, field, value in rows:
        try:
            verify_value(json.loads(raw), field, value)
        except ValueError as e:
            if value is not None:          # NULL = TED '-1' placeholder nulled on load (documented)
                bad.append(f"{nid}: {e}")
    return {"ted_notices": len(rows), "ted_awards": n_aw, "value_check_failures": bad}


def counts(path: Path) -> dict:
    import duckdb

    from src.db import TABLES

    with duckdb.connect(str(path), read_only=True) as con:
        out = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
        out["tenders_by_source"] = dict(con.execute("SELECT coalesce(source,'ted'), count(*) FROM tenders "
                                                    "GROUP BY 1 ORDER BY 1").fetchall())
        out["fts_window"] = [str(x) for x in con.execute(
            "SELECT min(publication_date), max(publication_date) FROM tenders WHERE source = 'uk_fts'").fetchone()]
        out["cells"] = {f"{c}/{m}": (lo, hi, st) for c, m, lo, hi, st in con.execute(
            "SELECT cluster, method, value_low, value_high, status FROM market_cells WHERE year = 2025").fetchall()}
    return out


def plausibility(s, fs, new_counts: dict) -> list[str]:
    """Checks that should hold for a sane build; each failure is a line."""
    from src.models.market_model import TOTAL

    out = []
    mk = s.cell(TOTAL, "reconciled")
    if not mk.low < mk.high:
        out.append(f"market range empty: {mk.low}..{mk.high}")
    if mk.floor_check == "floor above range":
        out.append("German register floor above the market range low")
    sh = s.ds.shares[TOTAL]
    if not 0.02 <= sh.low <= sh.high <= 0.40:
        out.append(f"DS share range {sh.low:.1%}-{sh.high:.1%} outside 2-40%")
    flagged = [c for c in s.share_checks if c.flag]
    out += [f"implied DS share flag: {c.method} {c.cluster} {c.flag} – {c.reason}" for c in flagged]
    for r in s.peers.results:
        if r.mc["total"][1] <= 0:
            out.append(f"{r.label}: restated total not positive")
    for c in s.peers.crosscheck:
        if not c.plausible:
            out.append(f"DACH cross-check failed: {c.peer} – {c.verdict}")
    o = s.model.market.supply["others_share"]
    if not 0.4 <= o["low"] <= o["high"] <= 0.95:
        out.append(f"others' share from tenders {o['low']:.0%}-{o['high']:.0%} outside 40-95%")
    for f in fs:
        for x in f.figures:
            if x.grade == "C" and x.text and "–" not in x.text and " to " not in x.text and "%" in x.text \
                    and not x.text.startswith((">", "+", "-")) and "notices" not in x.text and "draws" not in x.label:
                out.append(f"grade C figure not shown as a range: {x.label} = {x.text}")
            if x.grade in ("A", "B") and not x.sources:
                out.append(f"grade {x.grade} figure without source: {x.label}")
    for t in ("facts", "assumptions", "market_cells", "tenders", "tender_awards"):
        if not new_counts.get(t):
            out.append(f"table {t} is empty")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--swap", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in (out, Path(str(out) + ".wal")):
        p.unlink(missing_ok=True)
    os.environ["DUCKDB_PATH"] = str(out)          # every loader and model below reads/writes the new file
    from src.settings import get_settings
    get_settings.cache_clear()
    old = ROOT / "data" / "demo.duckdb"
    assert get_settings().duckdb_path == out

    from src.connectors import uk_fts
    from src.db import connect, init_db
    from src.facts_store import build_all, check_cells, check_rows, save

    print("1) schema");                      # noqa: E702
    with connect(out) as con:
        init_db(con)
    print("2) TED from the current DB (no file cache) + rule-1 re-check")
    ted = copy_ted(old, out)
    print(f"   {ted['ted_notices']} notices, {ted['ted_awards']} awards, "
          f"{len(ted['value_check_failures'])} value-check failures")
    print("3) UK FTS load-cache")
    fts = uk_fts.run_fetch(db_path=out, cache_only=True)
    print(f"   {fts}")
    print("4) facts / assumptions / market cells from caches")
    facts, assumptions, cells = build_all()
    check_rows(facts)
    check_cells(cells)
    res = save(facts, assumptions, db_path=out, market_cells=cells)
    print(f"   {res['facts']} facts, {res['assumptions']} assumptions, {res['market_cells']} cells")
    print("5) scenario, findings, report")
    from src.models import findings, scenario
    from src.models.others_share import load_awards
    from src.report.academia_report import OUT_DEFAULT, build

    base = scenario.prepare()
    s = scenario.run(base, {})
    awards = load_awards()
    fs = findings.build(s, base, awards)
    pdf = build(s, base, awards, fs)
    OUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DEFAULT.write_bytes(pdf)
    print(f"   report {OUT_DEFAULT} ({len(pdf) / 1024:,.0f} KB)")
    print("6) checks and diff")
    new_c = counts(out)
    problems = [f"TED value check: {x}" for x in ted["value_check_failures"]] + plausibility(s, fs, new_c)
    old_c = counts(old) if old.exists() else {}
    for k in sorted(set(new_c) | set(old_c)):
        if k == "cells":
            continue
        a, b = old_c.get(k), new_c.get(k)
        print(f"   {k:<18} old {a!s:<40} new {b!s:<40} {'' if a == b else '<- differs'}")
    for k in sorted(set(new_c.get("cells", {})) | set(old_c.get("cells", {}))):
        a, b = old_c.get("cells", {}).get(k), new_c.get("cells", {}).get(k)
        same = a and b and all((x == y) or (isinstance(x, float) and abs(x - y) < 0.05) for x, y in zip(a, b))
        if not same:
            print(f"   cell {k:<28} old {a} new {b}")
    print("   problems:" if problems else "   problems: none")
    for p in problems:
        print("    -", p)
    if args.swap:
        if problems:
            sys.exit("not swapping: checks failed")
        shutil.copy2(old, str(old) + ".bak")
        shutil.copy2(out, old)
        print(f"   swapped: {old} (backup {old}.bak)")


if __name__ == "__main__":
    main()
