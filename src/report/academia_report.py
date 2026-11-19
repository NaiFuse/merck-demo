"""Academia report as PDF: three headline findings, market range with the Merck anchor band, peer comparison,
tender win shares, source list and the 'With your data' page. Every number carries a grade and a source reference.

    uv run python -m src.report.academia_report [out.pdf]     # default: data/reports/academia_report.pdf

Built with reportlab (pure Python) from the same objects the dashboard uses (scenario, findings, tender awards);
no network, no DB writes. Fonts: Bitstream Vera (bundled with reportlab) for €, en dashes and minus signs.
"""

from __future__ import annotations

import argparse
import io
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import reportlab
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from xml.sax.saxutils import escape

from src.settings import PROJECT_ROOT

OUT_DEFAULT = PROJECT_ROOT / "data" / "reports" / "academia_report.pdf"
DISCLAIMER = ("Unofficial demo built from free public data only. Not endorsed by or affiliated with Merck KGaA, "
              "Darmstadt, Germany. Grades: A = disclosed, B = triangulated from 2+ sources, C = estimate (always shown "
              "as a range).")
CLUSTER_LABEL = {"dach": "DACH", "france_benelux": "France & Benelux", "uk_ireland_nordics": "UK, Ireland & Nordics",
                 "total": "3 clusters total"}
METHOD_NAME = {"A": "method A (supply side)", "B": "method B (demand side)",
               "merck_anchor": "the Merck anchor (broader scope)"}
GROUP_COLOR = {"thermo_fisher": "#2a78d6", "avantor": "#eb6834", "merck": "#1baf7a", "market": "#8a8984",
               "other": "#6250d6"}

_FONT_DIR = os.path.join(os.path.dirname(reportlab.__file__), "fonts")
pdfmetrics.registerFont(TTFont("Vera", os.path.join(_FONT_DIR, "Vera.ttf")))
pdfmetrics.registerFont(TTFont("VeraBd", os.path.join(_FONT_DIR, "VeraBd.ttf")))
pdfmetrics.registerFont(TTFont("VeraIt", os.path.join(_FONT_DIR, "VeraIt.ttf")))

S = {
    "title": ParagraphStyle("title", fontName="VeraBd", fontSize=18, leading=22, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="VeraBd", fontSize=13, leading=16, spaceBefore=8, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="VeraBd", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Vera", fontSize=8.5, leading=11.5, alignment=TA_LEFT, spaceAfter=3),
    "small": ParagraphStyle("small", fontName="Vera", fontSize=7, leading=9),
    "cell": ParagraphStyle("cell", fontName="Vera", fontSize=6.8, leading=8.4),
    "cellb": ParagraphStyle("cellb", fontName="VeraBd", fontSize=6.8, leading=8.4),
    "note": ParagraphStyle("note", fontName="VeraIt", fontSize=7.5, leading=9.5, textColor=colors.HexColor("#52514e")),
}


def P(text, style="body") -> Paragraph:
    return Paragraph(escape(str(text)), S[style])


class Sources:
    """Numbered source list: add() returns the reference number shown next to each figure."""

    def __init__(self):
        self.items: list[tuple[str, str, str]] = []
        self._idx: dict[tuple[str, str], int] = {}

    def add(self, ref: str, quote: str = "", grade: str = "") -> str:
        key = (ref, quote)
        if key not in self._idx:
            self.items.append((ref, quote, grade))
            self._idx[key] = len(self.items)
        return f"[{self._idx[key]}]"


def table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    data = [[P(c, "cellb" if (header and i == 0) else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")))
    t.setStyle(TableStyle(style))
    return t


def bullets(items: list[str], style="small") -> ListFlowable:
    return ListFlowable([ListItem(P(x, style), leftIndent=8) for x in items], bulletType="bullet", start="•",
                        leftIndent=8, bulletFontName="Vera", bulletFontSize=7)


def eur(x: float) -> str:
    return f"€{x / 1000:,.2f}bn" if abs(x) >= 1000 else f"€{x:,.0f}m"


def eur_rng(lo: float, hi: float) -> str:
    return f"€{lo / 1000:,.2f}–{hi / 1000:,.2f}bn" if hi >= 1000 else f"€{lo:,.0f}–{hi:,.0f}m"


def growth_drawing(f) -> Drawing:
    figs = [x for x in f.figures if x.value is not None]
    h = 14 * len(figs) + 36
    d = Drawing(170 * mm, h)
    ch = HorizontalBarChart()
    ch.x, ch.y, ch.width, ch.height = 92 * mm, 26, 70 * mm, h - 32
    ch.data = [[x.value * 100 for x in figs[::-1]]]
    ch.categoryAxis.categoryNames = [x.label for x in figs[::-1]]
    ch.categoryAxis.labels.fontName, ch.categoryAxis.labels.fontSize = "Vera", 6.5
    ch.categoryAxis.labels.boxAnchor = "e"
    ch.valueAxis.labels.fontName, ch.valueAxis.labels.fontSize = "Vera", 6.5
    lo = min(0, min(x.value for x in figs) * 100)
    vmin, vmax = np.floor(lo - 6), np.ceil(max(x.value for x in figs) * 100 + 4)   # room for labels left of negatives
    ch.valueAxis.valueMin, ch.valueAxis.valueMax = vmin, vmax
    ch.valueAxis.valueStep = 5
    # category labels sit at the zero line by default; move them left of the plot area
    ch.categoryAxis.labels.dx = -(0 - vmin) / (vmax - vmin) * ch.width - 3
    ch.valueAxis.labelTextFormat = "%+d%%"
    ch.bars.strokeWidth = 0
    for i, x in enumerate(figs[::-1]):
        ch.bars[(0, i)].fillColor = colors.HexColor(GROUP_COLOR.get(x.group, "#8a8984"))
    ch.barLabelFormat = lambda v: f"{v:+.1f}%"
    ch.barLabels.fontName, ch.barLabels.fontSize, ch.barLabels.nudge = "Vera", 6.5, 9
    d.add(ch)
    d.add(String(ch.x, 4, "growth vs. prior year, % (grade A; values in the table below)", fontName="Vera",
                 fontSize=6.5))
    return d


# ---------------------------------------------------------------- sections

def section_findings(story, fs, src: Sources) -> None:
    story.append(P("1  Headline findings", "h1"))
    for i, f in enumerate(fs, 1):
        block = [P(f"Finding {i}: {f.title}", "h2"), P(f.lead)]
        if f.key == "germany_channel":
            block.append(growth_drawing(f))
        rows = [["Figure", "Value", "Grade", "Period", "Source"]]
        for x in f.figures:
            refs = " ".join(src.add(s.ref, s.quote, s.grade) for s in x.sources) or src.add(
                "model: scenario Monte Carlo (10,000 draws), see section 5 assumptions", "", "C")
            rows.append([x.label, x.text, x.grade, x.period, refs])
        block.append(table(rows, [80, 32, 13, 20, 25]))
        block.append(Spacer(1, 3))
        block.append(P("How to read this", "cellb"))
        block.append(bullets(f.caveats))
        block.append(P(f.grade_note, "note"))
        story.append(KeepTogether(block[:3]))
        story.extend(block[3:])
        story.append(Spacer(1, 6))


def section_market(story, s, base, src: Sources) -> None:
    from src.models.market_model import ANCHOR_LABEL

    m = s.model.market
    o = m.supply["others_share"]
    story.append(PageBreak())
    story.append(P("2  Academia market, FY2025 (EUR m)", "h1"))
    story.append(P(f"Method A (supply side): Thermo Fisher + Avantor restated in 6 steps from SEC filings, grossed up by "
                   f"an others' share of {o['low']:.0%}–{o['high']:.0%} derived from TED + UK FTS win shares (grade "
                   f"{o['grade']}). Method B (demand side): academic R&D spend (Eurostat, OECD MSTI) × consumables "
                   f"intensity × (1 + non-R&D uplift). A cell is published as a point only if A and B agree within "
                   f"{s.model.tolerance:.0%}; otherwise the published grade-C range is the overlap of A and B, checked "
                   f"against the German register floor. Merck's own Discovery market (CMD 2025, scaled to the "
                   f"clusters) is shown separately – {ANCHOR_LABEL} – and is never part of the range."))
    r_model = src.add("model: market_model + reconciliation (P10–P90 of 10,000 draws); inputs in section 5", "", "C")
    rows = [["Cluster", "A (P10–P90)", "B (P10–P90)", "A/B gap", "Published range", "Grade", "German floor (P10)",
             "Merck anchor band (separate)", "Anchor gap to range", "Src"]]
    for k in list(m.clusters) + ["total"]:
        a, b, r = s.cell(k, "A"), s.cell(k, "B"), s.cell(k, "reconciled")
        fl = next((c for c in s.checks.get(k, []) if c.method == "de_floor"), None)
        band = r.bands.get("merck_anchor")
        rows.append([CLUSTER_LABEL[k], eur_rng(a.low, a.high), eur_rng(b.low, b.high), f"{r.gap_pct:.0%}",
                     eur_rng(r.low, r.high) + ("" if r.status == "range" else f" ({r.status})"), r.grade,
                     f"≥ {eur(fl.pct[0])}" if fl else "–",
                     eur_rng(band["low"], band["high"]) if band else "–",
                     (f"{band['gap_eur']:+,.0f} ({band['gap_pct']:+.0%})" if band else "–"), r_model])
    story.append(table(rows, [19, 21, 21, 11, 24, 13, 15, 22, 21, 13]))
    an = s.check_info.get("anchor")
    if an:
        f = an["fact"]
        ra = src.add(f"{f.url_or_path} p.14", f.quote, f.grade)
        story.append(P(f"Merck anchor = Merck's Discovery addressable market €{f.value:g}bn (2025, global, grade "
                       f"{f.grade}) {ra} × Europe share {an['europe_share'].low:.1%}–{an['europe_share'].high:.1%} "
                       f"(grade B) × cluster share (grade B) × Academia weight {an['academia_weight'].low:.0%}–"
                       f"{an['academia_weight'].high:.0%} × scope overlap {an['scope_overlap'].low:.0%}–"
                       f"{an['scope_overlap'].high:.0%} (grade C). Company-defined addressable markets are broad.",
                       "small"))
    fl = s.check_info.get("floor")
    if fl:
        de = np.percentile(fl["de"], (10, 90))
        story.append(P(f"German register floor (Germany only): {eur_rng(*de)} = Fisher Scientific GmbH + VWR "
                       f"International GmbH + Carl Roth ({fl['carl_roth_period']}), DS-type Academia purchases; a lower "
                       f"bound (excludes Merck's own sales and all other suppliers), scaled by Germany's research share "
                       f"for DACH and the total. Grade C.", "small"))
    sh = s.ds.shares["total"]
    ds = s.ds.academia_pct("total")
    story.append(Spacer(1, 4))
    story.append(table([["Figure", "Value", "Grade", "Source"],
                        ["DS Academia revenue, 3 clusters (estimate)", eur_rng(ds[0], ds[2]), s.ds.grade, r_model],
                        ["DS share of the Academia market", f"{sh.low:.0%}–{sh.high:.0%}", sh.grade, r_model]]
                       + [[f"DS share vs. {METHOD_NAME.get(k, k)}", f"{v[0]:.1%}–{v[2]:.1%}", "C", r_model]
                          for k, v in sh.vs.items()],
                       [80, 40, 14, 20]))


def section_peers(story, s, src: Sources) -> None:
    story.append(P("3  Peer comparison (DS-comparable EMEA Academia revenue, FY2025)", "h1"))
    rows = [["Company", "Grade"] + [CLUSTER_LABEL[k] for k in list(s.model.market.clusters) + ["total"]]
            + ["of which resold Merck", "Source"]]
    for r, cfg in zip(s.peers.results, s.peers.configs):
        ref = src.add(cfg.tenk_url, f"{r.label} {r.period} 10-K (SEC EDGAR XBRL facts) + config/assumptions/"
                                    f"{cfg.peer}.yaml; 6-step restatement", "A/C")
        rows.append([r.label, r.grade] + [eur_rng(r.mc[k][0], r.mc[k][2]) for k in list(s.model.market.clusters) + ["total"]]
                    + [eur_rng(r.mc["merck_total"][0], r.mc["merck_total"][2]), ref])
    ref = src.add("config/assumptions/merck_ds.yaml (Merck CMD 2025, Q3 2026 release) + model", "", "C")
    rows.append(["Merck DS (estimate)", s.ds.grade] + [eur_rng(*np.percentile(s.ds.academia[k], (10, 90)))
                                                        for k in list(s.model.market.clusters) + ["total"]] + ["–", ref])
    story.append(table(rows, [26, 13, 23, 23, 25, 23, 23, 14]))
    story.append(P("P10–P90 of 10,000 Monte Carlo draws. Output grade = worst input grade. Peer revenue keeps resold "
                   "Merck product (channel resale), shown separately.", "note"))
    for r in s.peers.results:
        rows = [["Step", "Factor (min / base / max)", "Value (min / base / max)", "Grade", "Detail"]]
        for x in r.steps:
            fac = " / ".join(f"{v:.1%}" for v in x.factor) if x.factor else ""
            rows.append([f"{x.n} {x.label}", fac, " / ".join(f"{v:,.0f}" for v in x.value), x.grade, x.detail])
        story.append(P(f"{r.label}: restatement steps (USD m; step 6 EUR m)", "h2"))
        story.append(table(rows, [38, 32, 34, 13, 53]))


def section_tenders(story, awards: pd.DataFrame, src: Sources) -> None:
    from src.models.tender_report import WIN_SHARE_LABEL, in_scope, win_shares
    from src.models.winners import GROUP_LABELS, PEER_GROUPS

    story.append(P("4  Tender win shares – Academia buyers (TED + UK FTS)", "h1"))
    story.append(P(f"{WIN_SHARE_LABEL}. Default view: core CPV, Academia buyers (universities, hospitals, research "
                   "organisations, government labs), excl. IVD winners, clinical blood-collection items, analyser "
                   "bundles and Merck lab water. Framework values are ceilings; multi-winner values split equally "
                   "(grade C). UK (Find a Tender) covered to January 2026."))
    ref = src.add("TED API (ted.europa.eu) + UK Find a Tender Service OCDS API; tables tenders / tender_awards",
                  "per-award source_url and evidence field in the database", "C")
    d = in_scope(awards)
    for title, x in (("All awards", d), ("Awards with a disclosed per-winner value only (grade A/B)",
                                         d[d["grade"].isin(["A", "B"])])):
        t = win_shares(x)
        t = t[[not str(p).startswith("2023") for p in t.index]]
        rows = [["Period"] + [f"{GROUP_LABELS[g]} value / awards" for g in PEER_GROUPS] + ["Total EUR m", "Grade",
                                                                                        "Source"]]
        for p, row in t.iterrows():
            rows.append([str(p)] + [f"{row[(GROUP_LABELS[g], 'value %')]:.1f}% / {row[(GROUP_LABELS[g], 'awards %')]:.1f}%"
                                    for g in PEER_GROUPS] + [f"{row[('total', 'EUR m')]:,.1f}", "C", ref])
        story.append(P(title, "h2"))
        story.append(table(rows, [22, 21, 21, 21, 22, 21, 16, 13, 14]))


def section_sources(story, s, base, src: Sources) -> None:
    from src.models.scenario import merck_assumptions

    story.append(PageBreak())
    story.append(P("5  Sources and assumptions", "h1"))
    story.append(P("Numbered sources referenced in sections 1–4. Disclosed figures carry the verbatim quote that the "
                   "code checked against the source (file + page, or URL)."))
    rows = [["#", "Source (file + page or URL)", "Quote / row", "Grade"]]
    for i, (ref, quote, grade) in enumerate(src.items, 1):
        rows.append([str(i), ref, quote[:600], grade])
    story.append(table(rows, [9, 62, 92, 13]))
    story.append(P("Model inputs (scenario values; grade C inputs are ranges)", "h2"))
    moved = set(s.overrides)
    rows = [["Owner.key", "Low / base / high", "Grade", "Source"]]
    items = s.assumptions() + [("merck_ds", a) for a in merck_assumptions(s, base)]
    for owner, a in items:
        fmt = (lambda v: f"{v:.1%}") if a.unit == "share" and a.high <= 1.5 else (lambda v: f"{v:,.4g}")
        rows.append([f"{owner}.{a.key}" + (" (slider)" if f"{owner}.{a.key}" in moved else ""),
                     f"{fmt(a.low)} / {fmt(a.base)} / {fmt(a.high)}", a.grade,
                     a.source[:300] + (f" [{a.source_file} p.{a.page}]" if a.source_file else "")])
    story.append(table(rows, [47, 34, 13, 82]))


def section_phase2(story) -> None:
    from src.app.views.phase2 import MODULES

    story.append(PageBreak())
    story.append(P("6  With your data (phase 2 – not active in this demo)", "h1"))
    story.append(P("Everything above comes from free public data. With Merck's internal data, the grade-C ranges for DS "
                   "become grade-A figures and the remaining uncertainty is the market itself."))
    rows = [["Module", "What it adds", "Needs"]]
    for title, _icon, text, needs in MODULES:
        rows.append([title, text, needs])
    story.append(table(rows, [32, 100, 42]))


# ---------------------------------------------------------------- build

def build(scenario, base, awards: pd.DataFrame, fs: list | None = None) -> bytes:
    from src.models import findings

    fs = fs or findings.build(scenario, base, awards)
    buf = io.BytesIO()
    scen = (f"Scenario: {len(scenario.overrides)} input(s) moved from the public-data default: "
            + ", ".join(f"{k} = {v:g}" for k, v in scenario.overrides.items())) if scenario.overrides else \
        "Scenario: public-data default (no slider moved)."

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Vera", 6.5)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(15 * mm, 9 * mm, "Unofficial demo · free public data · not endorsed by Merck KGaA, Darmstadt, "
                                           "Germany")
        canvas.drawRightString(195 * mm, 9 * mm, f"Merck DS EMEA Academia – outside-in report · page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title="Merck DS EMEA Academia – outside-in report",
                            author="merck-ds-demo (unofficial)")
    src = Sources()
    story = [P("Merck DS in EMEA Academia – outside-in report", "title"),
             P(f"DACH · France & Benelux · UK, Ireland & Nordics · FY2025 · generated {date.today():%d %b %Y}", "small"),
             Spacer(1, 4), P(DISCLAIMER, "note"), P(scen, "note"), Spacer(1, 6)]
    section_findings(story, fs, src)
    section_market(story, scenario, base, src)
    section_peers(story, scenario, src)
    section_tenders(story, awards, src)
    section_sources(story, scenario, base, src)
    section_phase2(story)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)
    from src.models import scenario as sc
    from src.models.others_share import load_awards

    base = sc.prepare()
    pdf = build(sc.run(base, {}), base, load_awards())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pdf)
    print(f"wrote {out} ({len(pdf) / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
