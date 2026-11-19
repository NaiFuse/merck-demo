"""Academia: Market (A vs B), Competitors (restated peers vs DS estimate), Customer proxies (tenders, funding, papers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app import ui
from src.models.market_model import TOTAL
from src.models.research_base import fill_interior

CL = list(ui.CLUSTERS) + [TOTAL]


# ---------------------------------------------------------------- market

CHECK_STYLE = {"merck_anchor": ("Merck's own addressable-market definition (broader scope)", "#6250d6"),
               "de_floor": ("German register floor (scaled, lower bound)", "#eda100")}


def market_block(ctx) -> None:
    s = ctx.s
    st.markdown(f"DS-type products bought by Academia, FY{s.year}, EUR m. **A** = supply side (Thermo Fisher + "
                f"Avantor restated, grossed up by the others' share from tender win shares); **B** = demand side "
                f"(academic R&D spend x consumables intensity x non-R&D uplift). Two independent checks: the **Merck "
                f"anchor** (Merck's own Discovery market, CMD 2025, scaled to the clusters – a broader scope, shown "
                f"as a separate band, never part of the range) and the **German register floor** (Fisher Scientific "
                f"GmbH + VWR International GmbH + Carl Roth, a lower-bound check). Published as a point only if A and "
                f"B agree within {s.model.tolerance:.0%}; otherwise the grey bar is the overlap of A and B.")
    rows, table = [], []
    for k in CL:
        a, b, r = s.cell(k, "A"), s.cell(k, "B"), s.cell(k, "reconciled")
        for m, c in (("A – supply side", a), ("B – demand side", b)):
            rows.append({"y": ui.CLUSTER_LABEL[k], "series": m, "p10": c.low, "p50": c.base, "p90": c.high,
                         "color": ui.COLOR[m[0]]})
        extra = {}
        for c in s.checks.get(k, []):
            label, color = CHECK_STYLE[c.method]
            q = c.pct
            rows.append({"y": ui.CLUSTER_LABEL[k], "series": label, "p10": q[0], "p50": q[1], "p90": q[2],
                         "color": color})
            extra["Merck anchor (separate)" if c.method == "merck_anchor" else "German floor"] = (
                f"≥ {ui.eur(q[0])}" if c.lower_only else ui.eur_range(q[0], q[2]))
        rows.append({"y": ui.CLUSTER_LABEL[k], "series": "Published range", "p10": r.low,
                     "p50": r.base if r.base is not None else (r.low + r.high) / 2, "p90": r.high, "color": "#8a8984"})
        table.append({"cluster": ui.CLUSTER_LABEL[k], "A": ui.eur_range(a.low, a.high),
                      "B": ui.eur_range(b.low, b.high), **extra, "A/B gap": f"{r.gap_pct:.0%}",
                      "published": ui.figure(r.low, r.base, r.high, r.grade), "status": r.status,
                      "floor check": r.floor_check or "–",
                      "anchor gap to range": (lambda g: "–" if not g else (
                          f"{'+' if g['gap_eur'] >= 0 else '−'}{ui.eur(abs(g['gap_eur']))} ({g['gap_pct']:+.0%})"))(
                          r.bands.get("merck_anchor")),
                      "grade": r.grade})
    st.plotly_chart(ui.range_chart(rows, "EUR m (P10–P90; P50 on hover)"), width="stretch")
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.markdown(ui.badge("C") + " :gray[all ranges P10–P90 of 10,000 draws; floor counts only as a lower bound]")

    st.markdown("##### Method B over time – 3 clusters total (EUR m)")
    dem = [c for c in s.model.demand if c.cluster == TOTAL]
    years = [c.year for c in dem]
    p = np.array([c.pct for c in dem])
    nowcast = [c.nowcast_share for c in dem]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years + years[::-1], y=list(p[:, 2]) + list(p[::-1, 0]), fill="toself",
                             fillcolor="rgba(235,104,52,0.18)", line=dict(width=0), name="P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years, y=p[:, 1], mode="lines+markers", name="P50", line=dict(color=ui.COLOR["B"], width=2),
                             marker=dict(size=8), customdata=np.c_[p[:, 0], p[:, 2], nowcast],
                             hovertemplate="%{x}: P10–P90 €%{customdata[0]:,.0f}–%{customdata[1]:,.0f}m"
                                           "<br>nowcast share of spend %{customdata[2]:.0%}<extra></extra>"))
    first_now = next((y for y, n in zip(years, nowcast) if n > 0.5), None)
    if first_now:
        fig.add_vrect(x0=first_now - 0.5, x1=years[-1] + 0.5, fillcolor="rgba(128,128,128,0.10)", line_width=0,
                      annotation_text="mostly nowcast", annotation_position="top left")
    fig.update_yaxes(title="EUR m", rangemode="tozero")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(ui.style(fig, 320), width="stretch")
    with c2:
        sp = pd.DataFrame([{"year": c.year, "cluster": ui.CLUSTER_LABEL[c.cluster], "spend": c.spend_raw}
                           for c in s.model.demand if c.cluster != TOTAL])
        st.markdown("**Driver: academic R&D spend** (HES + GOV + PNP, EUR m, Eurostat; UK OECD MSTI)")
        st.dataframe(sp.pivot(index="year", columns="cluster", values="spend").style.format("{:,.0f}"),
                     width="stretch")
        st.caption("Years after a series' last observation are nowcast at the cluster's 2019–23 CAGR.")


# ---------------------------------------------------------------- gap reconciliation

def reconciliation_block(ctx) -> None:
    from src.models.others_share import views

    s = ctx.s
    g = s.gap
    m = s.model.market
    o, prior = m.supply["others_share"], m.supply.get("others_share_prior")
    st.markdown("##### 1 · Others' share from tender win shares (method A)")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("Others' share, tenders", ui.pct_range(o["low"], o["high"]), border=True,
                  help="1 − (Thermo Fisher + Avantor) share of Academia tender awards; min–max over 8 views.")
        st.markdown(ui.badge(o["grade"], short=True)
                    + (f" :gray[prior free assumption {prior['low']:.0%}–{prior['high']:.0%}]" if prior else ""))
        if g.prior_gap is not None:
            st.caption(f"A/B gap at input medians: {g.prior_gap:+.0%} with the prior → {g.gap:+.0%} with tender data.")
    with c2:
        try:
            vs = views(ui.tenders())
            st.dataframe(pd.DataFrame([{"measure": v.measure, "frameworks": "incl." if v.frameworks else "excl.",
                                        "distributors": v.distributors, "awards": v.n_awards,
                                        "TMO + Avantor": f"{v.peers:.1%}", "distributors share": f"{v.distributor_share:.1%}",
                                        "others": f"{v.others:.1%}"} for v in vs]), hide_index=True, width="stretch")
        except Exception as e:  # noqa: BLE001
            ui.db_error(e)
    st.caption("Above-threshold public tenders only; small catalogue orders below the thresholds are not in the data.")

    st.markdown("##### 2 · What drives the A/B gap")
    st.markdown(f"All inputs at their median: A **{ui.eur(g.a)}** vs. B **{ui.eur(g.b)}**, signed gap "
                f"**{g.gap:+.0%}** ((B − A) / mean). Each bar moves one input across its stated range; the break-even "
                "is the value that alone would make A = B.")
    items = g.items[:12][::-1]
    fig = go.Figure()
    for i in items:
        inside = "#1baf7a" if i.in_range else "#8a8984"
        fig.add_trace(go.Scatter(
            x=[i.gap_low, i.gap_high], y=[f"{i.owner}.{i.key}"] * 2, mode="lines+markers", showlegend=False,
            line=dict(color=inside, width=8), marker=dict(size=9, color=inside),
            hovertemplate=(f"<b>{i.owner}.{i.key}</b><br>range {i.low:.4g} – {i.high:.4g}<br>gap at low "
                           f"{i.gap_low:+.0%}, at high {i.gap_high:+.0%}<br>break-even "
                           f"{'–' if i.break_even is None else f'{i.break_even:.4g}'}<extra></extra>")))
    fig.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.7)")
    fig.add_vline(x=g.gap, line_width=1, line_dash="dot", line_color="#e34948",
                  annotation_text=f"current {g.gap:+.0%}", annotation_position="top")
    fig.update_xaxes(title="signed A/B gap, (B − A) / mean", tickformat="+.0%", showgrid=True,
                     gridcolor="rgba(128,128,128,0.18)")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(ui.style(fig, 40 * len(items) + 90), width="stretch")
        st.caption("Green = the input can close the gap alone within its stated range.")
    with c2:
        st.dataframe(pd.DataFrame([{"input": f"{i.owner}.{i.key}", "range": f"{i.low:.4g} – {i.high:.4g}",
                                    "break-even": "–" if i.break_even is None else f"{i.break_even:.4g}",
                                    "within range": "yes" if i.in_range else "no", "grade": i.grade}
                                   for i in g.items[:12]]), hide_index=True, width="stretch")

    st.markdown("##### 3 · Implied DS share by method")
    rows = [{"method": c.method, "cluster": ui.CLUSTER_LABEL[c.cluster],
             "DS share (P10–P90)": ui.pct_range(c.share[0], c.share[2], 1),
             "flag": {"": "ok", "check": "⚠ check", "implausible": "✖ implausible"}[c.flag], "reason": c.reason}
            for c in s.share_checks]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    band = m.reconciliation.get("ds_share_plausible", (0.05, 0.30))
    st.caption(f"Implausible = DS alone exceeds the market minus Thermo Fisher and Avantor (plus the Merck product they "
               f"resell). Check = P50 outside the heuristic {band[0]:.0%}–{band[1]:.0%} band.")

    c1, c2 = st.columns(2)
    fl = s.check_info.get("floor")
    with c1:
        st.markdown("##### 4 · German register floor")
        if fl:
            de = np.percentile(fl["de"], (10, 50, 90))
            st.markdown(f"Germany, DS-type Academia purchases from three register filers: **{ui.eur_range(de[0], de[2])}**"
                        f" {ui.badge('C', True)}")
            st.dataframe(pd.DataFrame([{"company": k, "P10–P90": ui.eur_range(*np.percentile(v, (10, 90)))}
                                       for k, v in fl["parts"].items()]), hide_index=True, width="stretch")
            st.caption("A lower bound: excludes Merck's own sales, Thermo Fisher's direct entities and all other "
                       "suppliers. Scaled to DACH / the 3 clusters by Germany's share of academic research.")
        else:
            st.info(s.check_info.get("floor_error", "not available"))
    an = s.check_info.get("anchor")
    with c2:
        st.markdown("##### 5 · Merck market anchor")
        if an:
            f = an["fact"]
            st.markdown(f"Merck's own Discovery addressable market: **€{f.value:g}bn** (2025, global) "
                        f"{ui.badge(f.grade, True)}")
            st.markdown(f"× Europe {ui.pct_range(an['europe_share'].low, an['europe_share'].high, 1)} "
                        f"{ui.badge(an['europe_share'].grade, True)} × cluster share {ui.badge('B', True)} × Academia "
                        f"weight {ui.pct_range(an['academia_weight'].low, an['academia_weight'].high)} × scope overlap "
                        f"{ui.pct_range(an['scope_overlap'].low, an['scope_overlap'].high)} {ui.badge('C', True)}")
            st.caption("Europe share = min / median / max of " + ", ".join(f"{k} {v:.1%}" for k, v in
                                                                         an["europe_values"].items())
                       + ". Company-defined addressable markets tend to be broad – the anchor sits above A and B.")
        else:
            st.info(s.check_info.get("anchor_error", "not available"))


# ---------------------------------------------------------------- competitors

def competitors_block(ctx) -> None:
    s = ctx.s
    st.markdown(f"DS-comparable EMEA Academia revenue FY{s.year}, EUR m (P10–P90). Peers: 6-step restatement of "
                "SEC filings (*Peer comparison*). Merck DS: public-data estimate (sidebar).")
    rows, table = [], []
    for k in CL:
        for r in s.peers.results:
            q = r.mc[k]
            rows.append({"y": ui.CLUSTER_LABEL[k], "series": r.label, "p10": q[0], "p50": q[1], "p90": q[2],
                         "color": ui.COLOR[r.peer]})
        q = s.ds.academia_pct(k)
        rows.append({"y": ui.CLUSTER_LABEL[k], "series": "Merck DS (est.)", "p10": q[0], "p50": q[1], "p90": q[2],
                     "color": ui.COLOR["merck"]})
    for r in s.peers.results:
        tq, mq = r.mc["total"], r.mc["merck_total"]
        table.append({"company": r.label, "3 clusters (P10–P90)": ui.eur_range(tq[0], tq[2]),
                      "of which resold Merck": ui.eur_range(mq[0], mq[2]), "grade": r.grade,
                      "basis": f"{r.period} 10-K, restated"})
    q = s.ds.academia_pct(TOTAL)
    table.append({"company": "Merck DS (est.)", "3 clusters (P10–P90)": ui.eur_range(q[0], q[2]),
                  "of which resold Merck": "–", "grade": s.ds.grade, "basis": "public-data estimate"})
    a = s.cell(TOTAL, "A")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(ui.range_chart(rows, "EUR m (P10–P90; P50 on hover)"), width="stretch")
    with right:
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        others = s.model.market.supply["others_share"]
        st.caption(f"Peers keep resold Merck product in their revenue (channel resale), shown separately. "
                   f"Method A grosses the two peers up by an others' share of {others['low']:.0%}–{others['high']:.0%} "
                   f"(Merck direct, specialists, regional distributors) to {ui.eur_range(a.low, a.high)}.")


# ---------------------------------------------------------------- customer proxies

def tender_block() -> None:
    from src.models.tender_report import WIN_SHARE_LABEL, in_scope, win_shares
    from src.models.winners import GROUP_LABELS, PEER_GROUPS

    try:
        df = ui.tenders()
    except Exception as e:  # noqa: BLE001
        ui.db_error(e)
        return
    c1, c2, c3, c4 = st.columns(4)
    metric = c1.segmented_control("Measure", ["value %", "awards %"], default="value %", key="t_metric") or "value %"
    fw = c2.toggle("Include framework agreements (ceilings, not spend)", value=True, key="t_fw")
    bundles = c3.toggle("Include analyser bundles (sensitivity)", value=False, key="t_bund")
    disclosed = c4.toggle("Only awards with a disclosed per-winner value (grade A/B)", value=False, key="t_ab")
    d = in_scope(df, include_bundles=bundles)
    if not fw:
        d = d[~d["is_framework"]]
    by_grade = d.groupby("grade")["value_eur"].sum()
    if disclosed:
        d = d[d["grade"].isin(["A", "B"])]
    t = win_shares(d)
    t = t[[not str(p).startswith(("2023",)) for p in t.index]]   # 5-week stub period at the window start
    groups = [GROUP_LABELS[g] for g in PEER_GROUPS]
    fig = go.Figure()
    for gname in groups:
        fig.add_trace(go.Bar(x=[str(p) for p in t.index], y=t[(gname, metric)], name=gname,
                             marker=dict(color=ui.COLOR.get(gname, "#8a8984"), cornerradius=4),
                             hovertemplate=f"<b>{gname}</b> %{{x}}: %{{y:.1f}}%<extra></extra>"))
    fig.update_yaxes(title=metric.replace(" %", " share, %"), ticksuffix="%")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(ui.style(fig, 340, barmode="group", bargap=0.25, bargroupgap=0.08), width="stretch")
    with right:
        tot = t[("total", "EUR m")]
        n = t[("total", "awards")]
        st.dataframe(pd.DataFrame({"period": [str(p) for p in t.index], "EUR m awarded": tot.round(1).values,
                                   "awards": n.values}), hide_index=True, width="stretch")
        st.caption(f"{WIN_SHARE_LABEL}. Academia buyers only (universities, hospitals, research organisations, "
                   "government labs), core CPV, excl. IVD winners, clinical blood-collection items and Merck lab water. "
                   "UK (FTS) covered to Jan 2026 only.")
        tot_v = by_grade.sum() or 1.0
        st.markdown(ui.badge("C", short=True) + " :gray[value by award grade: "
                    + ", ".join(f"{g} {by_grade.get(g, 0) / tot_v:.0%}" for g in "ABC")
                    + " – C = multi-winner value split equally. Compare both toggle states for the range.]")
    with st.expander("Win share by cluster"):
        periods = [str(p) for p in t.index]
        per = st.selectbox("Period", periods, index=max(0, len(periods) - 2), key="t_per")
        dd = d[d["period"] == per]
        if len(dd):
            col = "value_eur" if metric == "value %" else "notice_id"
            agg = "sum" if metric == "value %" else "size"
            pv = dd.pivot_table(index="cluster", columns="group", values=col, aggfunc=agg, fill_value=0)
            pv = pv.reindex(columns=PEER_GROUPS, fill_value=0)
            out = pv.div(pv.sum(axis=1), axis=0).mul(100).rename(columns=GROUP_LABELS)
            st.dataframe(out.style.format("{:.1f}%"), width="stretch")


def _by_cluster(series_by_country: dict[str, dict[int, float]], y0: int, y1: int) -> pd.DataFrame:
    rows = []
    for k, label in ui.CLUSTERS.items():
        for c in ui.TAX["clusters"][k]["countries"]:
            for y, v in fill_interior(series_by_country.get(c, {})).items():
                if y0 <= y <= y1:
                    rows.append({"cluster": label, "year": y, "value": v})
    return pd.DataFrame(rows).groupby(["year", "cluster"], as_index=False)["value"].sum()


def _line(df: pd.DataFrame, y_title: str, kind: str = "line") -> go.Figure:
    fig = go.Figure()
    for k, label in ui.CLUSTERS.items():
        d = df[df["cluster"] == label]
        if kind == "bar":
            fig.add_trace(go.Bar(x=d["year"], y=d["value"], name=label, marker=dict(color=ui.CLUSTER_COLOR[k],
                                 cornerradius=4), hovertemplate=f"{label} %{{x}}: %{{y:,.0f}}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=d["year"], y=d["value"], name=label, mode="lines+markers",
                                     line=dict(color=ui.CLUSTER_COLOR[k], width=2), marker=dict(size=8),
                                     hovertemplate=f"{label} %{{x}}: %{{y:,.0f}}<extra></extra>"))
    fig.update_yaxes(title=y_title, rangemode="tozero")
    return ui.style(fig, 300, barmode="group")


def funding_publications_block(ctx) -> None:
    from src.connectors.eurostat import load_config
    from src.models.research_base import INDICATOR_LABEL, cluster_share_assumptions

    p = ctx.base.panel
    spend: dict[str, dict[int, float]] = {}
    for (c, _), sr in p.spend.items():
        for y, v in fill_interior(sr).items():
            spend.setdefault(c, {})[y] = spend.get(c, {}).get(y, 0.0) + v
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Funding: academic R&D spend** (HES + GOV + PNP, EUR m, observed years)")
        st.plotly_chart(_line(_by_cluster(spend, 2015, 2025), "EUR m"), width="stretch")
        st.caption("Eurostat rd_e_gerdtot; UK HES/GOV from OECD MSTI (GBP at ECB annual average). Grade A.")
    with c2:
        st.markdown("**Funding: EU research grants** to universities + research organisations (EUR m, CORDIS)")
        g = _by_cluster({c: {y: v / 1e6 for y, v in s.items()} for c, s in p.grants.items()}, 2014, 2025)
        st.plotly_chart(_line(g, "EUR m net EC contribution", "bar"), width="stretch")
        st.caption("H2020 + Horizon Europe by project start year. UK and CH were not associated to Horizon Europe "
                   "2021–23, so their recent grants drop. Grade A.")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Publications:** wet-lab papers from academia-type institutions (OpenAlex)")
        st.plotly_chart(_line(_by_cluster(p.publications, 2015, 2025), "works"), width="stretch")
        st.caption("Sum of the configured wet-lab topic queries (config/research_sources.yaml). Grade A counts.")
    with c4:
        st.markdown("**Cluster share of European academic research** – used to split peer revenue")
        shares, ind = cluster_share_assumptions(p, load_config(), ctx.base.market.research_base)
        tab = pd.DataFrame([{"indicator": INDICATOR_LABEL[i.indicator], "window": f"{i.window[0]}–{i.window[1]}",
                             **{ui.CLUSTERS[k]: f"{v:.1%}" for k, v in i.shares.items()}} for i in ind])
        tab.loc[len(tab)] = {"indicator": "→ used: base = R&D spend, range = min–max", "window": "",
                             **{ui.CLUSTERS[k]: f"{a.low:.1%}–{a.high:.1%}" for k, a in shares.items()}}
        st.dataframe(tab, hide_index=True, width="stretch")
        st.markdown(ui.badge("B"))


def render(ctx) -> None:
    st.title("Academia")
    st.caption("Universities, research institutes, hospitals and government labs (Merck counts hospitals as Academia).")
    with st.container(border=True):
        st.subheader("1 · Market")
        market_block(ctx)
    with st.container(border=True):
        st.subheader("1b · Gap reconciliation")
        reconciliation_block(ctx)
    with st.container(border=True):
        st.subheader("2 · Competitors")
        competitors_block(ctx)
    with st.container(border=True):
        st.subheader("3 · Customer proxies")
        st.markdown("##### Tender wins")
        tender_block()
        st.markdown("##### Funding and publications")
        funding_publications_block(ctx)
