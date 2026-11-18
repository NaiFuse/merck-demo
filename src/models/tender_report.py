"""Win share in public tenders above EU/UK thresholds (TED + UK FTS), by winner group, cluster, period.

    uv run python -m src.models.tender_report [--tier core|broad|all]

These are win shares in published above-threshold public tenders, NOT market shares.

Conventions
- Period = publication date, everywhere (the fetch window is defined on publication date, so full
  years are complete). Award dates are never mixed in.
- Default view = core CPV tier, excluding IVD/diagnostics winners, analyser bundles (analyser +
  installation/maintenance + reagents, title-based) and Merck awards tagged lab water. Each exclusion is
  reported separately; analyser bundles also as a sensitivity line.
- Win share is shown by value and by number of awards won (one award = one winner on one lot/contract).
- Framework-agreement values are ceilings, not spend. Shares are shown with and without them.
- Academia buyers only (config/academia_buyers.yaml, all sources): universities and university consortia,
  hospitals and hospital purchasing bodies, hospital-running health regions, research organisations, government
  labs. Removed value and notices are reported per source (table 0a).
- Framework-like notices (>= 5 suppliers, value split equally) count as frameworks.
- UK (FTS) coverage is the window of cached FTS months; periods running past it are marked
  'UK to <date>' (GB values there are missing, not zero).
"""

from __future__ import annotations

import argparse
import re
from datetime import date

import duckdb
import pandas as pd

from src.connectors.ted import load_cpv_config
from src.connectors.uk_fts import coverage as fts_coverage
from src.db import connect_readonly
from src.models.academia import classify_buyer
from src.models.scope import LAB_WATER, LAB_WATER_POSSIBLE
from src.models.taxonomy import load_taxonomy
from src.models.winners import GROUP_LABELS, IVD, OTHER, PEER_GROUPS, _fold, clean_name, winner_group

FRAMEWORK_LABEL = "framework (ceiling, not spend)"
WIN_SHARE_LABEL = "Win share in public tenders above EU/UK thresholds (not market share)"

# Titles that bundle analysers/instruments, installation or maintenance with reagents. Such notices sit
# under reagent CPVs but much of their value is instruments/service (excluded categories), so they are
# excluded from the default view and shown as a sensitivity line.
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
               t.value_eur AS notice_value_eur, t.currency, t.value AS notice_value,
               coalesce(t.source, 'ted') AS source, t.buyer_legal_type
        FROM tender_awards a JOIN tenders t USING (notice_id)
    """).df()
    # TED uses -1 as a 'value not disclosed' placeholder (seen on DE notices): treat as no value, keep the raw row
    df["value_withheld"] = (df["value_eur"] < 0) | (df["notice_value_eur"] < 0)
    df.loc[df["value_eur"] < 0, "value_eur"] = float("nan")
    df.loc[df["notice_value_eur"] < 0, "notice_value_eur"] = float("nan")
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
    df["is_framework_declared"] = df["is_framework"].fillna(False).astype(bool)
    like = framework_like_notices(df)
    df["framework_like"] = df["notice_id"].isin(like) & ~df["is_framework_declared"]
    df["is_framework"] = df["is_framework_declared"] | df["framework_like"]   # the 'frameworks' bucket
    df["instrument_bundle"] = df["title"].map(is_instrument_bundle)
    cls = {k: classify_buyer(*k) for k in
           df[["buyer", "buyer_legal_type", "source"]].drop_duplicates().itertuples(index=False, name=None)}
    keys = list(zip(df["buyer"], df["buyer_legal_type"], df["source"]))
    df["buyer_category"] = [cls[k].category for k in keys]
    df["academia"] = [cls[k].keep for k in keys]
    return df


FRAMEWORK_LIKE_MIN_SUPPLIERS = 5


def framework_like_notices(df: pd.DataFrame, min_suppliers: int = FRAMEWORK_LIKE_MIN_SUPPLIERS) -> set[str]:
    """Notices that behave like frameworks: one notice, >= min_suppliers distinct winners, and every valued award
    is an equal split of the notice value (alignment 'split_equal')."""
    g = df.groupby("notice_id").agg(
        suppliers=("winner_name", "nunique"),
        all_split=("alignment", lambda a: bool(len(a)) and set(a) <= {"split_equal", "no_value"} and "split_equal" in set(a)),
    )
    return set(g[(g["suppliers"] >= min_suppliers) & g["all_split"]].index)


COVERAGE_NOTES = [
    "Sources: TED (EU/EEA/CH) and UK Find a Tender Service (FTS). The UK left TED after Brexit; FTS is live "
    "since 1 Jan 2021, so UK above-threshold tenders are only on FTS.",
    "Below-threshold UK tenders are published on Contracts Finder, not FTS, and are not included; "
    "below-threshold EU tenders are likewise only on national portals.",
]
UK_COUNTRY = "GB"


def uk_coverage_notes(cov: dict | None, window_start: date, window_end: date) -> list[str]:
    """Lines describing the UK (FTS) coverage window, from the FTS month cache (uk_fts.coverage)."""
    if cov is None:
        return ["UK (FTS): no cached months – UK (GB) is not covered at all."]
    start, end = max(cov["start"], window_start), min(cov["end"], window_end)
    out = [f"UK (FTS) coverage window: publication dates {start:%d %b %Y} .. {end:%d %b %Y} "
           f"(cached months {cov['first_month']} .. {cov['last_month']}); TED covers {window_start:%d %b %Y} .. "
           f"{window_end:%d %b %Y}."]
    if end < window_end:
        out.append(f"GB figures after {end:%d %b %Y} are MISSING, not zero: periods marked "
                   f"'UK to {end:%d %b}' understate UK, Ireland & Nordics and the all-cluster totals.")
    if cov.get("missing_months"):
        out.append(f"UK months not cached (gaps): {', '.join(cov['missing_months'])}.")
    if cov.get("low_scan_months"):
        out.append(f"UK months with < half the median award releases scanned (median {cov['median_scanned']}), "
                   f"possibly incomplete paging: {', '.join(cov['low_scan_months'])}.")
    return out


def periods_beyond_uk(d: pd.DataFrame, cov_end: date | None) -> set[str]:
    """Periods containing publication dates after the UK coverage end."""
    if cov_end is None:
        return set(d["period"])
    return set(d.loc[d["publication_date"] > cov_end, "period"])


def mark_uk_coverage(table: pd.DataFrame, beyond: set[str], cov_end: date | None, uk_cluster: str | None = None) -> pd.DataFrame:
    """Add a 'UK coverage' column: rows whose period runs past the FTS window (only UK-cluster rows if given)."""
    t = table.copy()
    label = f"UK to {cov_end:%d %b %Y}" if cov_end else "UK not covered"
    periods = t.index.get_level_values("period")
    hit = periods.isin(list(beyond))
    if uk_cluster is not None:
        hit &= t.index.get_level_values("cluster") == uk_cluster
    t["UK coverage"] = ["" if not h else label for h in hit]
    return t


def value_by_country_source(d: pd.DataFrame, cluster: str) -> pd.DataFrame:
    """EUR m by country and source x period for one cluster (makes the UK/FTS contribution visible)."""
    c = d[d["cluster"] == cluster]
    p = c.pivot_table(index=["country", "source"], columns="period", values="value_eur", aggfunc="sum", fill_value=0) / 1e6
    return p.reindex(columns=[x for x in period_order(d) if x in p.columns])


def in_scope(df: pd.DataFrame, tier: str = "core", include_bundles: bool = False, academia: bool = True) -> pd.DataFrame:
    """Default view: tier, Academia buyers only, minus IVD winners, Merck lab water and (unless include_bundles)
    analyser bundles."""
    d = df if tier == "all" else df[df["tier"] == tier]
    d = d[(d["group"] != IVD) & (d["scope_note"] != LAB_WATER)]
    if academia and "academia" in d.columns:
        d = d[d["academia"]]
    return d if include_bundles else d[~d["instrument_bundle"]]


def filter_removal(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Value and notices removed by the Academia buyer filter, per source and buyer category."""
    removed = before[~before["notice_id"].isin(set(after["notice_id"]))]
    out = removed.groupby(["source", "buyer_category"]).agg(
        removed_eur_m=("value_eur", lambda x: x.sum() / 1e6), notices=("notice_id", "nunique"))
    tot = before.groupby("source").agg(before_eur_m=("value_eur", lambda x: x.sum() / 1e6),
                                       before_notices=("notice_id", "nunique"))
    summ = removed.groupby("source").agg(removed_eur_m=("value_eur", lambda x: x.sum() / 1e6),
                                         notices=("notice_id", "nunique")).join(tot, how="right").fillna(0)
    summ["removed % value"] = summ["removed_eur_m"] / summ["before_eur_m"] * 100
    summ["removed % notices"] = summ["notices"] / summ["before_notices"] * 100
    summ.index = pd.MultiIndex.from_product([summ.index, ["ALL removed"]], names=["source", "buyer_category"])
    return pd.concat([out, summ]).sort_index()


def kept_by_category(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["source", "buyer_category"]).agg(eur_m=("value_eur", lambda x: x.sum() / 1e6),
                                                       notices=("notice_id", "nunique"))


def framework_like_summary(d: pd.DataFrame) -> pd.DataFrame:
    """Framework-like notices (pass the pre-filter view to also see those the Academia filter removes)."""
    x = d[d["framework_like"]].groupby(["notice_id"]).agg(
        source=("source", "first"), buyer=("buyer", "first"), academia=("academia", "first"),
        suppliers=("winner_name", "nunique"), eur_m=("value_eur", lambda v: v.sum() / 1e6), title=("title", "first"))
    return x.sort_values("eur_m", ascending=False)


def cluster_group_shares(d: pd.DataFrame, cluster: str, group: str) -> pd.DataFrame:
    """One group's win share inside one cluster, by period: value % and awards %, all awards and excl. frameworks."""
    c = d[d["cluster"] == cluster]
    out = {}
    for label, sub in [("all awards", c), ("excl. frameworks", c[~c["is_framework"]])]:
        v = sub.pivot_table(index="period", columns="group", values="value_eur", aggfunc="sum", fill_value=0)
        n = sub.pivot_table(index="period", columns="group", values="notice_id", aggfunc="size", fill_value=0)
        for name, p in (("value %", v), ("awards %", n)):
            p = p.reindex(index=[x for x in period_order(d) if x in p.index])
            out[(label, name)] = (p.get(group, 0) / p.sum(axis=1).where(p.sum(axis=1) > 0) * 100)
        out[(label, "EUR m")] = v.sum(axis=1).reindex(index=[x for x in period_order(d) if x in v.index]) / 1e6
        out[(label, "awards")] = n.sum(axis=1).reindex(index=[x for x in period_order(d) if x in n.index])
    return pd.DataFrame(out)


def period_order(df: pd.DataFrame) -> list[str]:
    return list(df.sort_values("publication_date")["period"].drop_duplicates())


def value_by_cluster_period(d: pd.DataFrame) -> pd.DataFrame:
    """EUR m and award counts by cluster x period, value split into framework ceilings and other contracts."""
    d = d.assign(kind=d["is_framework"].map({True: FRAMEWORK_LABEL, False: "non-framework"}))
    p = d.pivot_table(index=["cluster", "period"], columns="kind", values="value_eur", aggfunc="sum", fill_value=0) / 1e6
    p = p.reindex(columns=["non-framework", FRAMEWORK_LABEL], fill_value=0)
    p["total EUR m"] = p.sum(axis=1)
    g = d.groupby(["cluster", "period"])
    p["awards"] = g.size()
    p["notices"] = g["notice_id"].nunique()
    clusters = [v["label"] for v in load_taxonomy()["clusters"].values()]
    idx = pd.MultiIndex.from_product([clusters, period_order(d)], names=["cluster", "period"])
    p = p.reindex(idx, fill_value=0)
    p[["awards", "notices"]] = p[["awards", "notices"]].fillna(0).astype(int)
    return p


def group_shares(d: pd.DataFrame, metric: str = "value") -> pd.DataFrame:
    """% by winner group per period. metric='value' (EUR) or 'count' (awards won)."""
    if metric == "value":
        p = d.pivot_table(index="period", columns="group", values="value_eur", aggfunc="sum", fill_value=0)
    else:
        p = d.pivot_table(index="period", columns="group", values="notice_id", aggfunc="size", fill_value=0)
    p = p.reindex(index=period_order(d), columns=PEER_GROUPS, fill_value=0)
    total = p.sum(axis=1)
    out = p.div(total.where(total > 0), axis=0).mul(100).rename(columns=GROUP_LABELS)
    if metric == "value":
        out["total EUR m"] = total / 1e6
    else:
        out["total awards"] = total.astype(int)
    return out


def win_shares(d: pd.DataFrame) -> pd.DataFrame:
    """Value share and award-count share side by side: columns (group, 'value %' | 'awards %') + totals."""
    v, c = group_shares(d, "value"), group_shares(d, "count")
    cols = {}
    for g in PEER_GROUPS:
        lab = GROUP_LABELS[g]
        cols[(lab, "value %")] = v[lab]
        cols[(lab, "awards %")] = c[lab]
    cols[("total", "EUR m")] = v["total EUR m"]
    cols[("total", "awards")] = c["total awards"]
    return pd.DataFrame(cols)


def top_others(d: pd.DataFrame, n: int = 15, by: str = "value_eur_m") -> pd.DataFrame:
    """Top 'Others' winners by value or by award count, with the countries they won in and the (unverified)
    Merck/Sigma-Aldrich distributor mark from config/others_distributors.yaml."""
    o = d[d["group"] == OTHER].assign(company=lambda x: x["company"].replace("", "(winner name missing)"))
    g = o.groupby("company").agg(value_eur_m=("value_eur", lambda s: s.sum() / 1e6),
                                 awards=("notice_id", "size"),
                                 notices=("notice_id", "nunique"),
                                 example_name=("winner_name", "first"))
    by_country = o.groupby(["company", "country"])["value_eur"].sum().fillna(0)
    cnt_country = o.groupby(["company", "country"]).size()
    def countries(c):
        v = by_country.loc[c] if c in by_country.index.get_level_values(0) else pd.Series(dtype=float)
        k = cnt_country.loc[c]
        order = sorted(k.index, key=lambda x: (-(v.get(x, 0) or 0), -k[x]))
        return ", ".join(order)
    g["countries"] = [countries(c) for c in g.index]
    marks = load_distributor_marks()
    g["merck_sigma_distributor"] = [marks.get(c, {}).get("sells_merck_sigma", "not reviewed") for c in g.index]
    g["type"] = [marks.get(c, {}).get("type", "") for c in g.index]
    cols = ["value_eur_m", "awards", "notices", "countries", "type", "merck_sigma_distributor", "example_name"]
    return g.sort_values([by, "value_eur_m"], ascending=False).head(n)[cols]


def load_distributor_marks() -> dict:
    """{cleaned company name: {type, sells_merck_sigma, basis}} – review list, does not change winner mappings."""
    import yaml
    from src.settings import PROJECT_ROOT
    path = PROJECT_ROOT / "config" / "others_distributors.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("companies", {})


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
    wh = df[df["value_withheld"]] if "value_withheld" in df.columns else df.iloc[:0]
    if len(wh):
        out.append(f"{len(wh)} awards on {wh['notice_id'].nunique()} notices carry the placeholder value -1 "
                   f"('not disclosed'); treated as no value: e.g. {sorted(wh['notice_id'].unique())[:5]}")
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
    before_filter = in_scope(df, args.tier, academia=False)
    with_bundles = in_scope(df, args.tier, include_bundles=True)
    bundles = with_bundles[with_bundles["instrument_bundle"]]
    ivd = tier_df[tier_df["group"] == IVD]
    lw = tier_df[tier_df["scope_note"] == LAB_WATER]
    lwp = tier_df[tier_df["scope_note"] == LAB_WATER_POSSIBLE]

    by_src = df.drop_duplicates("notice_id")["source"].value_counts().to_dict()
    win_start, win_end = df["publication_date"].min(), df["publication_date"].max()
    cov = fts_coverage()
    cov_end = min(cov["end"], win_end) if cov else None
    beyond = periods_beyond_uk(df, cov_end) if (cov_end is None or cov_end < win_end) else set()
    print(f"Tender award notices {by_src} | tier={args.tier} | period basis: publication date "
          f"({win_start} .. {win_end})")
    for note in COVERAGE_NOTES + uk_coverage_notes(cov, win_start, win_end):
        print(f"Coverage: {note}")
    print(f"Default view (core CPV, Academia buyers, excl. IVD, analyser bundles, Merck lab water): "
          f"{d['notice_id'].nunique()} notices, {len(d)} awards, EUR {d['value_eur'].sum()/1e6:,.1f}m")
    print(f"Excluded – non-Academia buyers (schools, utilities, municipalities, general purchasing, unclassified): "
          f"{before_filter['notice_id'].nunique() - d['notice_id'].nunique()} notices, "
          f"EUR {(before_filter['value_eur'].sum() - d['value_eur'].sum())/1e6:,.1f}m (table 0a)")
    fl = d[d["framework_like"]]
    print(f"Framework-like (>= {FRAMEWORK_LIKE_MIN_SUPPLIERS} suppliers, value split equally, not declared a framework): "
          f"{fl['notice_id'].nunique()} notices, EUR {fl['value_eur'].sum()/1e6:,.1f}m – counted in the frameworks bucket (table 0c)")
    print(f"Excluded – {GROUP_LABELS[IVD]}: {ivd['notice_id'].nunique()} notices, EUR {ivd['value_eur'].sum()/1e6:,.1f}m")
    print(f"Excluded – analyser bundles (sensitivity): {bundles['notice_id'].nunique()} notices, {len(bundles)} awards, "
          f"EUR {bundles['value_eur'].sum()/1e6:,.1f}m (would add {bundles['value_eur'].sum()/max(d['value_eur'].sum(), 1):.0%} "
          f"to the default view)")
    print(f"Excluded – Merck lab water: {len(lw)} awards tagged, EUR {lw['value_eur'].sum()/1e6:,.2f}m; "
          f"{len(lwp)} more 'possible' (kept in scope, lot not attributable)")
    print(f"Note: '{FRAMEWORK_LABEL}' values are maximum contract values, not actual purchases.\n")

    print("0a) Academia buyer filter – removed vs. pre-filter default view, per source and buyer category\n",
          filter_removal(before_filter, d), "\n")
    print("0b) Kept buyers by category (EUR m, notices)\n", kept_by_category(d), "\n")
    print("0c) Framework-like notices (frameworks bucket); pre-filter view, academia=False ones are removed by 0a\n",
          framework_like_summary(before_filter), "\n")
    uk_cluster = load_taxonomy()["clusters"]["uk_ireland_nordics"]["label"]
    print("1) Default view: value and awards by cluster and period (EUR m)\n",
          mark_uk_coverage(value_by_cluster_period(d), beyond, cov_end, uk_cluster), "\n")
    by_cs = value_by_country_source(d, uk_cluster)
    by_cs.columns = [f"{c} [UK to {cov_end:%d %b}]" if c in beyond else c for c in by_cs.columns]
    print(f"1b) {uk_cluster} by country and source (EUR m)\n", by_cs, "\n")
    print("1c) Sensitivity – analyser bundles by cluster and period (EUR m, not in default view)\n",
          mark_uk_coverage(value_by_cluster_period(bundles)[["total EUR m", "awards", "notices"]], beyond, cov_end,
                           uk_cluster), "\n")
    for title, sub in [(f"2a) {WIN_SHARE_LABEL} – all awards (framework values are ceilings)", d),
                       (f"2b) {WIN_SHARE_LABEL} – excluding framework agreements", d[~d["is_framework"]]),
                       (f"2c) Sensitivity – {WIN_SHARE_LABEL} incl. analyser bundles, all awards", with_bundles)]:
        t = win_shares(sub)
        t[("", "UK coverage")] = [f"UK to {cov_end:%d %b %Y}" if p in beyond else "" for p in t.index]
        print(title + "\n", t, "\n")
    uk = d[d["country"] == UK_COUNTRY]
    print(f"2d) UK (GB, FTS) only – {WIN_SHARE_LABEL}, all awards\n", win_shares(uk) if len(uk) else "no UK awards", "\n")
    if len(uk):
        print(f"2e) UK (GB, FTS) only – excluding framework agreements\n",
              win_shares(uk[~uk["is_framework"]]) if (~uk["is_framework"]).any() else "all UK awards are frameworks", "\n")
    dach = load_taxonomy()["clusters"]["dach"]["label"]
    print(f"2f) Merck win share in {dach} only (value % / awards %; frameworks incl. framework-like)\n",
          cluster_group_shares(d, dach, "merck"), "\n")
    print("3a) Top 25 'Others' winners by value, default view (EUR m) – distributor mark is a review list, unverified\n",
          top_others(d, 25, "value_eur_m"), "\n")
    print("3b) Top 25 'Others' winners by award count, default view\n", top_others(d, 25, "awards"), "\n")
    print("4) Value by grade, default view (EUR m)\n", value_by_grade(d), "\n")
    print("5) Plausibility flags")
    for line in plausibility_checks(tier_df, d) or ["none"]:
        print("  -", line)


if __name__ == "__main__":
    main()
