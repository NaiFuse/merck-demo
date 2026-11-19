"""The three headline findings (Overview panels and the PDF report), every figure with grade and source.

    uv run python -m src.models.findings show

1  germany_channel    Thermo Fisher's German distributor grew academic sales +16.8% in 2025; VWR Germany declined;
                      DS +1.7% (Q3 2026, global). Disclosed figures side by side – no causal claim.
2  tender_radar       DS's catalogue model runs below the tender radar: >90% of DS customers spend < EUR 100k a year
                      (CMD), below the EU supply-contract thresholds; Merck is a direct winner of a small share of
                      Academia tender value while its estimated market share is far higher.
3  growth_bridge      Share bridge under two market-growth views (our nowcast vs Merck's ~LSD).
All arithmetic is plain Python on stored facts / model draws. Derived growth rates are written back as facts
(derived_fact_rows) so the database holds every number the Overview shows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.facts_store import FactRow

PROVENANCE_FILES = {"fisher_scientific_gmbh": "Fisher Scientific GmbH", "vwr_international_gmbh": "VWR International GmbH",
                    "carl_roth": "Carl Roth GmbH + Co. KG", "merck_kgaa": "Merck KGaA"}


@dataclass
class Source:
    ref: str              # 'data/raw/... p.3' or URL
    quote: str
    grade: str


@dataclass
class Figure:
    label: str
    value: float | None   # for charts (growth as a fraction); None for text-only figures
    text: str             # display text (ranges for grade C)
    grade: str
    sources: list[Source] = field(default_factory=list)
    period: str = ""
    group: str = ""       # chart grouping / colour key


@dataclass
class Finding:
    key: str
    title: str
    lead: str
    figures: list[Figure]
    caveats: list[str]
    grade_note: str


# ---------------------------------------------------------------- fact lookup

def fact(facts: list[FactRow], entity: str, period: str, metric: str, segment: str | None = None,
         region: str | None = None) -> FactRow:
    hits = [r for r in facts if r.entity == entity and r.period == period and r.metric == metric
            and (segment is None or r.segment == segment) and (region is None or r.region == region)]
    if not hits:
        raise KeyError(f"no fact {entity} {period} {metric} segment={segment} region={region}")
    return hits[0]


def src(r: FactRow) -> Source:
    page = r.quote.split(":", 1)[0] if r.quote.startswith("p.") else ""
    ref = f"{r.url_or_path} {page}".strip()
    return Source(ref, r.quote.split(": ", 1)[1] if page else r.quote, r.grade)


def pct_fig(r: FactRow, label: str, group: str) -> Figure:
    v = r.value / 100
    return Figure(label, v, f"{r.value:+.1f}%" if abs(r.value) >= 1 else f"{r.value:+.2f}%",
                  r.grade, [src(r)], r.period, group)


def _eur_m(r: FactRow) -> float:
    from src.extract.numbers import unit_multiplier
    return r.value * unit_multiplier(r.unit) / 1e6


# ---------------------------------------------------------------- derived growth facts

def de_rd_growth(facts: list[FactRow], year: int = 2024, sectors=("HES", "GOV", "PNP")) -> tuple[float, list[FactRow]]:
    rows = [r for r in facts if r.entity == "DE" and r.metric == "rd_expenditure" and r.segment in sectors
            and r.period in (str(year - 1), str(year)) and r.source == "eurostat"]
    by = {(r.segment, int(r.period)): r for r in rows}
    if len(by) != 2 * len(sectors):
        raise KeyError(f"DE R&D facts missing for {sectors} {year - 1}/{year}")
    cur = sum(by[(s, year)].value for s in sectors)
    prev = sum(by[(s, year - 1)].value for s in sectors)
    return cur / prev - 1, [by[k] for k in sorted(by)]


def derived_fact_rows(facts: list[FactRow]) -> list[FactRow]:
    """Growth rates computed from stored facts (plain Python), stored as derived facts with their inputs."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out = []
    for sectors, seg in ((("HES", "GOV", "PNP"), "HES+GOV+PNP"), (("HES",), "HES")):
        g, inp = de_rd_growth(facts, 2024, sectors)
        v = round(g * 100, 2)
        flags = sorted({r.quote.rsplit("(flag ", 1)[1].rstrip(")") for r in inp if "(flag " in r.quote})
        out.append(FactRow("derived", "derived:findings", now, "DE", "2024", "academic_rd_spend_growth", v, "%",
                           "DE", seg, f"derived: findings Eurostat rd_e_gerdtot {seg} DE 2024 vs 2023 = {v:.2f} % "
                           f"(EUR m {sum(r.value for r in inp if r.period == '2024'):,.1f} / "
                           f"{sum(r.value for r in inp if r.period == '2023'):,.1f})"
                           + (f"; input flags {','.join(flags)} (b = break in series)" if flags else ""), "A"))
    cur = fact(facts, "vwr_international_gmbh", "FY2025", "revenue", region="Deutschland")
    prev = fact(facts, "vwr_international_gmbh", "FY2024", "revenue", region="Deutschland")
    v = round((_eur_m(cur) / _eur_m(prev) - 1) * 100, 2)
    out.append(FactRow("derived", "derived:findings", now, "vwr_international_gmbh", "FY2025", "revenue_growth", v, "%",
                       "Deutschland", "total", f"derived: findings VWR International GmbH revenue Deutschland FY2025 vs "
                       f"FY2024 = {v:.2f} % (EUR m {_eur_m(cur):,.1f} / {_eur_m(prev):,.1f})", "A"))
    cur, prev = fact(facts, "carl_roth", "FY2024", "revenue", region="Inland"), fact(facts, "carl_roth", "FY2023",
                                                                                     "revenue", region="Inland")
    v = round((cur.value / prev.value - 1) * 100, 2)
    out.append(FactRow("derived", "derived:findings", now, "carl_roth", "FY2024", "revenue_growth", v, "%", "Inland",
                       "total", f"derived: findings Carl Roth revenue Inland FY2024 vs FY2023 = {v:.2f} % "
                       f"(EUR m {cur.value / 1e6:,.1f} / {prev.value / 1e6:,.1f})", "A"))
    return out


# ---------------------------------------------------------------- finding 1

def germany_channel(facts: list[FactRow], merck) -> Finding:
    d = {r.entity + r.period + r.metric + (r.segment or "") + (r.region or ""): r for r in derived_fact_rows(facts)}
    figs = [pct_fig(fact(facts, "fisher_scientific_gmbh", y, "segment_revenue_growth", "Academic"),
                    f"Fisher Scientific GmbH – Academic segment {y}", "thermo_fisher") for y in ("FY2023", "FY2024", "FY2025")]
    figs.append(pct_fig(fact(facts, "fisher_scientific_gmbh", "FY2025", "revenue_growth", "total"),
                        "Fisher Scientific GmbH – total revenue FY2025", "thermo_fisher"))
    figs.append(pct_fig(fact(facts, "vwr_international_gmbh", "FY2024", "revenue_growth",
                             "statement: reasons for the decline"), "VWR International GmbH – total revenue FY2024 (≈)",
                        "avantor"))
    figs.append(pct_fig(fact(facts, "vwr_international_gmbh", "FY2025", "revenue_growth", "total"),
                        "VWR International GmbH – total revenue FY2025", "avantor"))
    vd = d["vwr_international_gmbhFY2025revenue_growthtotalDeutschland"]
    figs.append(Figure("VWR International GmbH – revenue in Germany FY2025", vd.value / 100, f"{vd.value:+.1f}%", "A",
                       [Source("derived from data/raw/registers/de_vwr-international-gmbh_2025_jahresabschluss.pdf p.13",
                               vd.quote, "A")], "FY2025", "avantor"))
    g = merck.disclosed["ds_organic_growth_q3_2026"]
    figs.append(Figure("Merck Discovery Solutions – organic growth Q3 2026 (global)", g.base, f"{g.base * 100:+.1f}%",
                       g.grade, [Source(f"{g.source_file} p.{g.page}", g.quote, g.grade)], "Q3 2026", "merck"))
    for seg, label in (("HES", "German university R&D spend (HES) 2024"),
                       ("HES+GOV+PNP", "German academic R&D spend (HES+GOV+PNP) 2024")):
        r = d["DE2024academic_rd_spend_growth" + seg + "DE"]
        figs.append(Figure(label, r.value / 100, f"{r.value:+.1f}%", "A",
                           [Source("Eurostat rd_e_gerdtot (derived)", r.quote, "A")], "2024", "market"))
    cr = d["carl_rothFY2024revenue_growthtotalInland"]
    figs.append(Figure("Carl Roth – revenue in Germany FY2024 (latest filed)", cr.value / 100, f"{cr.value:+.1f}%", "A",
                       [Source("derived from data/raw/registers/de_carl-roth-gmbh-co-kg_2024_jahresabschluss.pdf",
                               cr.quote, "A")], "FY2024", "other"))
    reason = fact(facts, "vwr_international_gmbh", "FY2024", "revenue_growth", "statement: reasons for the decline")
    decl = fact(facts, "vwr_international_gmbh", "FY2025", "revenue_change")
    emp = {y: fact(facts, "fisher_scientific_gmbh", y, "employees",
                   segment="average number of employees" if y == "FY2024" else "average").value for y in ("FY2024", "FY2025")}
    vwr_tot = fact(facts, "vwr_international_gmbh", "FY2025", "revenue")
    vwr_abroad = [r for r in facts if r.entity == "vwr_international_gmbh" and r.period == "FY2025"
                  and r.metric == "revenue" and r.region == "Ausland"][0]
    abroad = _eur_m(vwr_abroad) / _eur_m(vwr_tot)
    lead = ("Thermo Fisher's German distributor Fisher Scientific GmbH grew academic sales +16.8% in 2025 (after +7.0% "
            "and +7.6%). VWR International GmbH, Avantor's German lab company, declined (about −4% in 2024, −0.59% in "
            "2025, the decline 'mainly from sales in Germany'). Merck DS grew +1.7% organically (Q3 2026, global). "
            "German academic R&D spend rose +7.1% in 2024.")
    caveats = [
        "Different scopes: Fisher Scientific GmbH and VWR International GmbH are single German legal entities; the DS "
        "figure is global Discovery Solutions, all customer groups.",
        "Different periods: register figures are fiscal years 2023–2025; DS growth is Q3 2026 vs Q3 2025.",
        "Fisher Scientific GmbH is a distributor: its academic sales include third-party brands, possibly Merck products.",
        f"Fisher's headcount fell ({emp['FY2024']:.0f} to {emp['FY2025']:.0f} average employees, FY2024 to FY2025) "
        "while revenue rose; volume may have moved from other Thermo Fisher entities in Germany – the filings do not say.",
        f"VWR gives no academic split; {abroad:.0%} of its FY2025 revenue is foreign (incl. intercompany). Its FY2024 '−4%' is "
        "'rund' (approximately) in the board minutes and cites 'Zurückhaltung der öffentlichen Hand bei "
        "Forschungsausgaben' (restraint in public research spending).",
        "Segment definitions ('Academic') are the companies' own and are not disclosed in detail.",
        "German R&D spend 2024 is the latest Eurostat year (2025 not yet published); GOV includes non-life-science labs.",
        "Nothing here shows a share shift between suppliers – these are disclosed growth rates side by side.",
    ]
    return Finding("germany_channel",
                   "Thermo Fisher's German distributor grew academic sales +16.8% in 2025; VWR Germany declined; "
                   "DS +1.7% (Q3 2026, global)", lead, figs, caveats,
                   f"All figures grade A (disclosed or computed from disclosed facts). Sources: register filings, "
                   f"Eurostat, Merck release. VWR statement: \"{reason.quote.split(': ', 1)[-1][:120]}…\"; "
                   f"decline TEUR {abs(decl.value):,.0f}.")


# ---------------------------------------------------------------- finding 2

def tender_shares(awards: pd.DataFrame) -> dict:
    """Pooled 2024 .. latest win shares by group: value (all awards / grade A+B only) and count."""
    from src.models.tender_report import in_scope

    d = in_scope(awards)
    d = d[~d["period"].astype(str).str.startswith("2023")]
    out = {}
    for name, x in (("all", d), ("ab", d[d["grade"].isin(["A", "B"])])):
        v, c = x.groupby("group")["value_eur"].sum(), x.groupby("group").size()
        out[name] = {"value": (v / v.sum()).to_dict(), "count": (c / c.sum()).to_dict(), "total_eur_m": v.sum() / 1e6}
    m = d[d["group"] == "merck"]
    out["merck_notices"], out["merck_eur_m"] = int(m["notice_id"].nunique()), float(m["value_eur"].sum() / 1e6)
    out["periods"] = sorted(d["period"].astype(str).unique())
    by_year = d[d["period"].astype(str).str.fullmatch(r"\d{4}")].groupby("period")["value_eur"].sum() / 1e6
    out["by_year_eur_m"] = by_year.to_dict()
    return out


def tender_radar(facts: list[FactRow], awards: pd.DataFrame, scenario) -> Finding:
    from src.models.market_model import TOTAL

    cmd = fact(facts, "merck_kgaa", "FY2025", "ds_customers_share_spend_below_threshold")
    cut = fact(facts, "merck_kgaa", "FY2025", "ds_customer_spend_cutoff")
    thr = {(r.period, r.metric): r for r in facts if r.source == "eurlex"}
    c26, s26 = thr[("2026", "procurement_threshold_supplies_central")], thr[("2026", "procurement_threshold_supplies_subcentral")]
    c25, s25 = thr[("2025", "procurement_threshold_supplies_central")], thr[("2025", "procurement_threshold_supplies_subcentral")]
    t = tender_shares(awards)

    def rng(group, metric="value"):
        a, b = t["all"][metric].get(group, 0.0), t["ab"][metric].get(group, 0.0)
        lo, hi = sorted([a, b])
        return f"{lo * 100:.1f}–{hi * 100:.1f}%"

    sh = scenario.ds.shares[TOTAL]
    mk = scenario.cell(TOTAL, "reconciled")
    y = max(t["by_year_eur_m"]) if t["by_year_eur_m"] else None
    cov = (t["by_year_eur_m"][y] / mk.high, t["by_year_eur_m"][y] / mk.low) if y else None
    tender_src = Source("TED + UK FTS award notices, default Academia view (tenders / tender_awards tables)",
                        "1 award = 1 winner on one lot/contract; multi-winner values split equally (grade C)", "C")
    figs = [
        Figure("DS customers spending < €100k a year (global, all DS customers)", cmd.value / 100, f">{cmd.value:.0f}%",
               cmd.grade, [src(cmd)], "CMD 2025", "merck"),
        Figure("EU threshold, supply contracts – central government (2026–27)", None, f"€{c26.value:,.0f}", "A",
               [Source(c26.url_or_path, c26.quote, "A")], "2026", "eu"),
        Figure("EU threshold, supply contracts – sub-central authorities (2026–27)", None, f"€{s26.value:,.0f}", "A",
               [Source(s26.url_or_path, s26.quote, "A")], "2026", "eu"),
        Figure("EU thresholds 2024–25 (central / sub-central)", None, f"€{c25.value:,.0f} / €{s25.value:,.0f}", "A",
               [Source(c25.url_or_path, c25.quote, "A")], "2024–25", "eu"),
        Figure(f"Merck direct win share of Academia tender value, {t['periods'][0]}–{t['periods'][-1][:4]} pooled", None,
               rng("merck"), "C", [tender_src], "2024–26", "merck"),
        Figure("Merck direct win share by number of awards", None, rng("merck", "count"), "C", [tender_src], "2024–26",
               "merck"),
        Figure("Merck direct wins", None, f"{t['merck_notices']} notices, €{t['merck_eur_m']:,.1f}m", "C", [tender_src],
               "2024–26", "merck"),
        Figure("Thermo Fisher win share of Academia tender value", None, rng("thermo_fisher"), "C", [tender_src],
               "2024–26", "thermo_fisher"),
        Figure("Avantor win share of Academia tender value", None, rng("avantor"), "C", [tender_src], "2024–26",
               "avantor"),
        Figure("DS estimated share of the Academia market (3 clusters, FY2025)", None,
               f"{sh.low * 100:.0f}–{sh.high * 100:.0f}%", sh.grade,
               [Source("model: DS Academia estimate ÷ reconciled market range (Sources & assumptions)", mk.note, "C")],
               "FY2025", "merck"),
    ]
    if cov:
        figs.append(Figure(f"Academia tender value {y} as share of the market range", None,
                           f"{cov[0] * 100:.0f}–{cov[1] * 100:.0f}%", "C",
                           [Source("tender awards ÷ reconciled market range", f"€{t['by_year_eur_m'][y]:,.0f}m awarded "
                                   f"in {y} vs market €{mk.low:,.0f}–{mk.high:,.0f}m", "C")], y, "market"))
    lead = (f"More than {cmd.value:.0f}% of DS customers spend less than €{cut.value:.0f}k a year with Merck (CMD 2025) – "
            f"below the EU threshold for a tendered supply contract (€{c26.value:,.0f} central government, "
            f"€{s26.value:,.0f} sub-central, 2026). Merck is the direct winner of {rng('merck')} of Academia tender "
            f"value (2024–26), against an estimated {sh.low * 100:.0f}–{sh.high * 100:.0f}% share of the Academia market: "
            "most DS academic business does not show up in public tenders.")
    caveats = [
        "The EU threshold applies to the buyer's estimated total contract value, not to its spend with one supplier: "
        "a university buying < €100k from Merck may still buy through a larger tendered framework won by a distributor.",
        "The CMD statement covers all DS customers worldwide (not Academia or EMEA only) and is a lower bound ('>90%').",
        "Tender win shares are shares of published above-threshold awards, not market shares; framework values are "
        "ceilings, multi-winner values are split equally (grade C); range = all awards vs awards with a disclosed "
        "per-winner value.",
        "UK (Find a Tender) data end in January 2026; UK thresholds differ from the EU's and are not shown; "
        "below-threshold national portals are not covered.",
        "The DS market share is a public-data estimate (grade C) against a market range that is itself grade C.",
    ]
    return Finding("tender_radar", "DS's catalogue model runs below the tender radar", lead, figs, caveats,
                   "CMD statement and EU thresholds grade A; tender win shares and DS share grade C (ranges).")


# ---------------------------------------------------------------- finding 3

def growth_bridge(scenario, merck) -> Finding:
    br = scenario.bridge
    figs = []
    for v in br.views:
        d = br.views[v]
        q = {k: np.percentile(d[k], (10, 50, 90)) for k in ("g_m", "mix", "share", "g_ds")}
        lab = br.view_labels[v]
        figs += [
            Figure(f"{lab}: market growth", None, f"{q['g_m'][0] * 100:.1f}–{q['g_m'][2] * 100:.1f}%", "C", [], "2025", v),
            Figure(f"{lab}: mix effect", None, f"{q['mix'][0] * 100:+.1f} to {q['mix'][2] * 100:+.1f}pp", "C", [], "", v),
            Figure(f"{lab}: share effect", None, f"{q['share'][0] * 100:+.1f} to {q['share'][2] * 100:+.1f}pp", "C", [],
                   "", v),
            Figure(f"{lab}: share of draws with a negative share effect", None, f"{float((d['share'] < 0).mean()):.0%}",
                   "C", [], "", v),
        ]
    g = merck.disclosed["ds_organic_growth_q3_2026"]
    cagr = merck.anchor.get("market_cagr", {})
    figs.insert(0, Figure("DS growth used in both views (Q3 2026 organic, global, as EMEA proxy)", g.base,
                          f"{g.base * 100:+.1f}% (range {merck.bridge['ds_growth']['low'] * 100:.1f}–"
                          f"{merck.bridge['ds_growth']['high'] * 100:.1f}%)", "C",
                          [Source(f"{g.source_file} p.{g.page}", g.quote, "A")], "Q3 2026", "merck"))
    m, l = (br.views[k] for k in ("model", "merck_lsd")) if "merck_lsd" in br.views else (br.draws, None)
    ms = np.percentile(m["share"], (10, 90))
    lead = (f"Against our 2025 market nowcast DS's share effect is {ms[0] * 100:+.1f} to {ms[1] * 100:+.1f}pp; ")
    if l is not None:
        ls = np.percentile(l["share"], (10, 90))
        lead += (f"against Merck's own '~LSD' view of the Discovery market it is {ls[0] * 100:+.1f} to "
                 f"{ls[1] * 100:+.1f}pp. Whether DS is losing share depends on the market view; mix explains almost "
                 "nothing in either.")
    caveats = [
        "Not like-for-like: DS growth is Q3 2026 vs Q3 2025 for global DS (EMEA proxy); market growth is 2025 vs 2024.",
        "Our market growth = method B nowcast (EMEA academic R&D spend extended at its 2019–23 CAGR) plus estimates for "
        "Pharma & Biotech and Industrial & Testing. German academic R&D spend grew faster in 2024 (+7.1%), so the "
        "nowcast may understate market growth.",
        f"Merck's view: CMD 2025 p.14 'Discovery ~LSD' (low single digit), translated to "
        f"{cagr.get('low', 0.01) * 100:.0f}–{cagr.get('high', 0.03) * 100:.0f}% (grade C); global, all customer groups.",
        "The share effect is a residual (DS growth − market growth − mix); DS growth by segment would split it.",
    ]
    return Finding("growth_bridge", "Is DS losing share? It depends on the market view", lead, figs, caveats,
                   "All bridge figures grade C (estimates, shown as P10–P90 ranges).")


def build(scenario, base, awards: pd.DataFrame) -> list[Finding]:
    return [germany_channel(base.facts, base.merck), tender_radar(base.facts, awards, scenario),
            growth_bridge(scenario, base.merck)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["show"])
    ap.parse_args(argv)
    from src.models import scenario as sc
    from src.models.others_share import load_awards

    base = sc.prepare()
    for f in build(sc.run(base, {}), base, load_awards()):
        print(f"\n## {f.title}\n{f.lead}\n  [{f.grade_note}]")
        for x in f.figures:
            print(f"  {x.text:>22}  [{x.grade}]  {x.label}" + (f"  <- {x.sources[0].ref}" if x.sources else ""))
        for c in f.caveats:
            print(f"   - {c}")


if __name__ == "__main__":
    main()
