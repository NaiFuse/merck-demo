"""TED award value by winner group, cluster and period (read-only).

    uv run python -m src.models.tender_report [--tier core|broad|all]

Conventions
- Period = TED publication date, everywhere (the fetch window is defined on publication date, so full
  years are complete). Award dates are never mixed in.
- Default scope = core CPV tier, excluding IVD/diagnostics winners and Merck awards tagged lab water.
  Both exclusions are reported separately.
- Framework-agreement values are ceilings, not spend. Shares are shown with and without them.
"""

from __future__ import annotations

import argparse
import re
from datetime import date

import duckdb
import pandas as pd

from src.connectors.ted import load_cpv_config
from src.db import connect_readonly
from src.models.scope import LAB_WATER, LAB_WATER_POSSIBLE
from src.models.taxonomy import load_taxonomy
from src.models.winners import GROUP_LABELS, IVD, OTHER, PEER_GROUPS, _fold, clean_name, winner_group

FRAMEWORK_LABEL = "framework (ceiling, not spend)"

# Titles that bundle analysers/instruments, installation or maintenance with reagents. Such notices sit
# under reagent CPVs but much of their value is instruments/service (excluded categories). Flag only.
INSTRUMENT_BUNDLE_RE = re.compile(
    r"automate|analyseur|analy[sz]er|analysator|equipement|installation|mise en service|location|maintenance|"
    r"gerat|instrument|automation|automaatio|imagerie|imaging|mikroskop|microscop|sequenceur|sequencer|"
    r"cytometr|zytometr|spectrometr|chromatograph",
)


def is_instrument_bundle(title) -> bool:
    return isinstance(title, str) and bool(INSTRUMENT_BUNDLE_RE.search(_fold(title)))


def period_label(d: date, window_start: date, window_end: date) -> str:
    if d.year == window_start.year and window_start > date(d.year, 1, 1):
        return f"{d.year} partial (from {window_start:%d %b})"
    if d.year == window_end.year and window_end < date(d.year, 12, 31):
        return f"{d.year} YTD (to {window_end:%d %b})"
    return str(d.year)


def load_awards(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT a.notice_id, a.seq, a.winner_name, a.value_eur, a.alignment, a.grade, a.scope_note, a.evidence,
               t.country, t.publication_date, t.cpv, t.is_framework, t.buyer, t.title,
               t.value_eur AS notice_value_eur, t.currency, t.value AS notice_value
        FROM tender_awards a JOIN tenders t USING (notice_id)
    """).df()
    tax = load_taxonomy()
    c2cl = {c: v["label"] for v in tax["clusters"].values() for c in v["countries"]}
    cfg = load_cpv_config()
    core = {c for c, v in cfg["include"].items() if v["tier"] == "core"}
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.date
    start, end = df["publication_date"].min(), df["publication_date"].max()
    df["period"] = df["publication_date"].map(lambda d: period_label(d, start, end))
    df["cluster"] = df["country"].map(c2cl)
    df["tier"] = df["cpv"].fillna("").str.split(";").map(lambda cs: "core" if core & set(cs) else "broad")
    df["group"] = df["winner_name"].map(winner_group)
    df["company"] = df["winner_name"].map(clean_name)
    df["is_framework"] = df["is_framework"].fillna(False).astype(bool)
    df["instrument_bundle"] = df["title"].map(is_instrument_bundle)
    return df


def in_scope(df: pd.DataFrame, tier: str = "core") -> pd.DataFrame:
    d = df if tier == "all" else df[df["tier"] == tier]
    return d[(d["group"] != IVD) & (d["scope_note"] != LAB_WATER)]


def period_order(df: pd.DataFrame) -> list[str]:
    return list(df.sort_values("publication_date")["period"].drop_duplicates())


def value_by_cluster_period(d: pd.DataFrame) -> pd.DataFrame:
    """EUR m by cluster x period, split into framework ceilings and other contracts."""
    d = d.assign(kind=d["is_framework"].map({True: FRAMEWORK_LABEL, False: "non-framework"}))
    p = d.pivot_table(index=["cluster", "period"], columns="kind", values="value_eur", aggfunc="sum", fill_value=0) / 1e6
    p = p.reindex(columns=["non-framework", FRAMEWORK_LABEL], fill_value=0)
    p["total"] = p.sum(axis=1)
    clusters = [v["label"] for v in load_taxonomy()["clusters"].values()]
    idx = pd.MultiIndex.from_product([clusters, period_order(d)], names=["cluster", "period"])
    return p.reindex(idx, fill_value=0)


def group_shares(d: pd.DataFrame) -> pd.DataFrame:
    p = d.pivot_table(index="period", columns="group", values="value_eur", aggfunc="sum", fill_value=0)
    p = p.reindex(index=period_order(d), columns=PEER_GROUPS, fill_value=0)
    total = p.sum(axis=1)
    out = p.div(total, axis=0).mul(100).rename(columns=GROUP_LABELS)
    out["total EUR m"] = total / 1e6
    return out


def top_others(d: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    o = d[d["group"] == OTHER].assign(company=lambda x: x["company"].replace("", "(winner name missing)"))
    g = o.groupby("company").agg(value_eur_m=("value_eur", lambda s: s.sum() / 1e6),
                                 notices=("notice_id", "nunique"),
                                 example_name=("winner_name", "first"))
    return g.nlargest(n, "value_eur_m")


def value_by_grade(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["grade", "alignment"]).agg(value_eur_m=("value_eur", lambda s: s.sum() / 1e6),
                                              rows=("notice_id", "size"))
    g["share %"] = g["value_eur_m"] / g["value_eur_m"].sum() * 100
    return g


def plausibility_checks(df: pd.DataFrame, d: pd.DataFrame) -> list[str]:
    """Flags for manual review. df = all awards, d = in-scope awards."""
    out = []
    total = d["value_eur"].sum()
    notices = d.groupby("notice_id").agg(v=("value_eur", "sum"), buyer=("buyer", "first"), title=("title", "first"),
                                         fw=("is_framework", "first"), country=("country", "first"))
    big = notices[notices["v"] > 0.03 * total].sort_values("v", ascending=False)
    for nid, r in big.iterrows():
        out.append(f"Single notice >3% of in-scope value: {nid} EUR {r.v/1e6:,.1f}m ({r.v/total:.0%}), "
                   f"{r.country}, framework={r.fw}, buyer={str(r.buyer)[:60]!r}, title={str(r.title)[:70]!r}")
    nv = df.drop_duplicates("notice_id")
    missing_val = nv["notice_value"].isna().sum()
    if missing_val:
        out.append(f"{missing_val} of {len(nv)} notices ({missing_val/len(nv):.0%}) have no award value at all")
    no_fx = nv[nv["notice_value"].notna() & nv["notice_value_eur"].isna()]
    if len(no_fx):
        out.append(f"{len(no_fx)} notices have a value but no EUR conversion (currencies: {sorted(no_fx['currency'].dropna().unique())})")
    tiny = nv[(nv["notice_value_eur"] > 0) & (nv["notice_value_eur"] < 1000)]
    if len(tiny):
        out.append(f"{len(tiny)} notices valued under EUR 1,000 (possible unit prices): e.g. {list(tiny['notice_id'][:5])}")
    huge = nv[nv["notice_value_eur"] > 50e6]
    for _, r in huge.iterrows():
        out.append(f"Notice over EUR 50m: {r.notice_id} EUR {r.notice_value_eur/1e6:,.0f}m, {r.country}, title={str(r.title)[:70]!r}")
    # names that look like a peer but fell into 'other' / IVD
    sus = df[(df["group"].isin([OTHER, IVD]))
             & df["winner_name"].fillna("").str.lower().str.contains(r"fisher|thermo|sigma|millipore|vwr|avantor|merck")]
    for name in sus["winner_name"].drop_duplicates()[:10]:
        out.append(f"Peer keyword but not mapped to a peer: {name!r} -> {winner_group(name)}")
    # XML vs search totals far apart
    xml = df[df["alignment"] == "xml"].groupby("notice_id").agg(x=("value_eur", "sum"), n=("notice_value_eur", "first"))
    off = xml[(xml["n"] > 0) & ((xml["x"] / xml["n"] - 1).abs() > 0.25)]
    for nid, r in off.iterrows():
        out.append(f"XML lot sum differs >25% from notice value: {nid} XML EUR {r.x/1e6:,.2f}m vs notice EUR {r.n/1e6:,.2f}m")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="core", choices=["core", "broad", "all"])
    args = ap.parse_args(argv)
    with connect_readonly() as con:
        df = load_awards(con)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 180, "display.max_rows", 200, "display.max_colwidth", 50,
                  "display.float_format", "{:,.1f}".format)
    tier_df = df if args.tier == "all" else df[df["tier"] == args.tier]
    d = in_scope(df, args.tier)
    ivd = tier_df[tier_df["group"] == IVD]
    lw = tier_df[tier_df["scope_note"] == LAB_WATER]
    lwp = tier_df[tier_df["scope_note"] == LAB_WATER_POSSIBLE]

    print(f"TED award notices | tier={args.tier} | period basis: publication date "
          f"({df['publication_date'].min()} .. {df['publication_date'].max()})")
    print(f"In scope: {d['notice_id'].nunique()} notices, EUR {d['value_eur'].sum()/1e6:,.1f}m")
    print(f"Excluded – {GROUP_LABELS[IVD]}: {ivd['notice_id'].nunique()} notices, EUR {ivd['value_eur'].sum()/1e6:,.1f}m")
    print(f"Excluded – Merck lab water: {len(lw)} awards tagged, EUR {lw['value_eur'].sum()/1e6:,.2f}m; "
          f"{len(lwp)} more 'possible' (kept in scope, lot not attributable)")
    print(f"Note: '{FRAMEWORK_LABEL}' values are maximum contract values, not actual purchases.\n")

    print("1) In-scope value by cluster and period (EUR m)\n", value_by_cluster_period(d), "\n")
    print("2a) Share by winner group, % of value – all notices (framework values are ceilings)\n", group_shares(d), "\n")
    print("2b) Share by winner group, % of value – excluding framework agreements\n", group_shares(d[~d["is_framework"]]), "\n")
    nb = d[~d["instrument_bundle"]]
    bundle_v = d.loc[d["instrument_bundle"], "value_eur"].sum()
    print(f"2c) Share by winner group – excluding instrument/analyser bundles (title-based flag; "
          f"bundles = EUR {bundle_v/1e6:,.1f}m, {bundle_v/max(d['value_eur'].sum(), 1):.0%} of in-scope value)\n",
          group_shares(nb), "\n")
    print("2d) Share by winner group – excluding frameworks AND instrument bundles\n",
          group_shares(nb[~nb["is_framework"]]), "\n")
    print("3) Top 15 'others' winners (EUR m)\n", top_others(d), "\n")
    print("4) Value by grade (EUR m)\n", value_by_grade(d), "\n")
    print("5) Plausibility flags")
    for line in plausibility_checks(tier_df, d) or ["none"]:
        print("  -", line)


if __name__ == "__main__":
    main()
