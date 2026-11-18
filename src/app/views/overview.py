"""Overview: headline finding (DS Academia revenue, market, share, tender wins) and the share bridge."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app import ui
from src.models.merck_ds import TOTAL


def tender_headline() -> dict | None:
    """Win share by value for the last full year, default view (Academia buyers, core CPV, all awards)."""
    from src.models.tender_report import in_scope, win_shares

    try:
        d = in_scope(ui.tenders())
    except Exception as e:  # noqa: BLE001
        ui.db_error(e)
        return None
    t, t_ab = win_shares(d), win_shares(d[d["grade"].isin(["A", "B"])])
    full = [p for p in t.index if str(p).isdigit() and p in t_ab.index]
    if not full:
        return None
    year = full[-1]

    def rng(g):   # grade C: range over (all awards, awards with a disclosed per-winner value only)
        v = sorted([t.loc[year, (g, "value %")], t_ab.loc[year, (g, "value %")]])
        return f"{v[0]:.1f}–{v[1]:.1f}%"

    return {"year": year, "merck": rng("Merck"), "tmo": rng("Thermo Fisher"), "avantor": rng("Avantor"),
            "total": t.loc[year, ("total", "EUR m")]}


def bridge_chart(br) -> go.Figure:
    q = {k: br.pct(br.draws[k]) for k in ("g_m", "mix", "share", "g_ds")}
    m = {k: v[1] for k, v in q.items()}
    steps = [("Market growth<br><sub>2025 nowcast</sub>", 0.0, m["g_m"], q["g_m"], ui.COLOR["Others"]),
             ("Mix effect", m["g_m"], m["mix"], q["mix"], "#6250d6"),
             ("Share effect<br><sub>residual</sub>", m["g_m"] + m["mix"], m["share"], q["share"],
              "#e34948" if m["share"] < 0 else "#1baf7a"),
             ("DS growth<br><sub>Q3 2026, global proxy</sub>", 0.0, m["g_ds"], q["g_ds"], ui.COLOR["Merck"])]
    fig = go.Figure()
    for name, base, h, (lo, _, hi), color in steps:
        label = ui.pp_range(lo, hi) if name.startswith(("Mix", "Share")) else ui.pct_range(lo, hi, 1)
        fig.add_trace(go.Bar(
            x=[name], y=[h], base=[base], marker=dict(color=color, cornerradius=4), width=0.55, showlegend=False,
            text=[label], textposition="outside", cliponaxis=False,
            hovertemplate=f"<b>{name}</b><br>P10–P90 {label}<br>P50 {h * 100:+.2f}pp<extra></extra>"))
    fig.update_yaxes(title="growth, %", tickformat=".1%")
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    return ui.style(fig, 380, bargap=0.35)


def render(ctx) -> None:
    s = ctx.s
    st.title("Merck DS in EMEA Academia – outside-in view")
    st.caption("DACH · France & Benelux · UK, Ireland & Nordics · FY2025 · free public data only · "
               "unofficial, not endorsed by Merck")

    ds = s.ds.academia_pct(TOTAL)
    mk = s.cell(TOTAL, "reconciled")
    a, b = s.cell(TOTAL, "A"), s.cell(TOTAL, "B")
    sh = s.ds.shares[TOTAL]
    th = tender_headline()

    c = st.columns(4)
    with c[0]:
        st.metric("DS Academia revenue, 3 clusters", ui.eur_range(ds[0], ds[2]), border=True,
                  help="Public-data estimate: DS EMEA revenue x Academia share x cluster share (P10–P90).")
        st.markdown(ui.badge(s.ds.grade, short=True))
    with c[1]:
        st.metric("DS-type Academia market", ui.figure(mk.low, mk.base, mk.high, mk.grade), border=True,
                  help="Published as a point only if supply side (A) and demand side (B) agree within 15%. Otherwise "
                       "the grade-C range is the overlap of A, B, the Merck anchor and the German register floor "
                       "(largest subset that overlaps).")
        st.markdown(ui.badge(mk.grade, short=True))
    with c[2]:
        st.metric("DS share of that market", ui.figure(sh.low, sh.base, sh.high, sh.grade, "pct"), border=True)
        st.markdown(ui.badge(sh.grade, short=True))
    with c[3]:
        if th:
            st.metric(f"Merck tender win share {th['year']}", th["merck"], border=True,
                      help="Share of award value in above-threshold public tenders by Academia buyers (TED + UK FTS). "
                           "Win share, not market share. Range: all awards vs. only awards with a disclosed "
                           "per-winner value (multi-winner values are otherwise split equally, grade C).")
            st.markdown(ui.badge("C", short=True) + " :gray[win share, not market share]")

    gap = mk.gap_pct or 0.0
    status = ("published (A and B agree within 15%)" if mk.status == "published"
              else f"A and B still differ by {gap:.0%} at the P50, so only a range is published")
    rec_line = reconciliation_line(s)
    lines = [
        f"**Merck DS sells an estimated {ui.eur_range(ds[0], ds[2])} of DS-type products to Academia** "
        f"(universities, institutes, hospitals, government labs) in the three clusters, out of "
        f"{ui.eur_range(ui_pct(s.ds.emea, 10), ui_pct(s.ds.emea, 90))} DS revenue in EMEA.",
        f"**The market is {ui.figure(mk.low, mk.base, mk.high, mk.grade)}**: supply side (peers restated + others) "
        f"{ui.eur_range(a.low, a.high)} vs. demand side (R&D spend x consumables intensity) {ui.eur_range(b.low, b.high)} "
        f"– {status}. {rec_line}",
        f"**That puts DS at {ui.figure(sh.low, sh.base, sh.high, sh.grade, 'pct')} of the Academia market** ("
        + ", ".join(f"{ui.pct_range(v[0], v[2])} against {METHOD_LABEL.get(m, m)}" for m, v in sh.vs.items()) + ").",
    ]
    if th:
        lines.append(f"**In public tenders Merck wins {th['merck']} of Academia award value** ({th['year']}, "
                     f"€{th['total']:,.0f}m awarded) vs. Thermo Fisher {th['tmo']} and Avantor {th['avantor']}: "
                     "likely because catalogue purchases fall below tender thresholds or go through distributors, "
                     "so tenders understate Merck's reach.")
    br = s.bridge
    g = {k: br.pct(br.draws[k]) for k in ("g_m", "mix", "share", "g_ds")}
    lines.append(f"**Growth: DS {ui.pct_range(g['g_ds'][0], g['g_ds'][2], 1)} vs. market "
                 f"{ui.pct_range(g['g_m'][0], g['g_m'][2], 1)}** – mix {ui.pp_range(g['mix'][0], g['mix'][2])}, "
                 f"share {ui.pp_range(g['share'][0], g['share'][2])}. :orange[Not like-for-like:] DS = Q3 2026 vs. "
                 "Q3 2025, global DS as EMEA proxy; market = 2025 nowcast (bridge below).")
    with st.container(border=True):
        st.markdown("#### Headline finding")
        for line in lines:
            st.markdown(f"- {line}")
        if ctx.s.overrides:
            st.caption(f":orange[Scenario: {len(ctx.s.overrides)} input(s) moved in the sidebar.] "
                       "Figures differ from the public-data default.")

    st.subheader("Share bridge – why DS grows differently from its market")
    st.latex(r"g_{DS}-g_M=\underbrace{\textstyle\sum_s (w_s-v_s)\,g_s}_{\text{mix}}"
             r"+\underbrace{\textstyle\sum_s w_s\,(g_{DS,s}-g_s)}_{\text{share}}")
    st.warning(
        "**Period mismatch – not a like-for-like comparison.** "
        "**DS growth** = Q3 2026 vs. Q3 2025 organic growth of *global* Discovery Solutions (+1.7%, disclosed), "
        "used as a proxy for DS EMEA. **Market growth** = 2025 vs. 2024: the Academia part is the method B *nowcast* "
        "(EMEA academic R&D spend extended at its 2019–23 CAGR), Pharma & Biotech and Industrial & Testing are "
        "estimates. Read the bridge as a direction indicator, not a measured share loss.", icon=":material/schedule:")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(bridge_chart(br), width="stretch", theme="streamlit")
        cagr = ctx.base.merck.anchor.get("market_cagr")
        st.caption("Bars at the P50 (median of 10,000 draws); labels show P10–P90. All bridge inputs are grade C. "
                   "Bar labels: DS growth = Q3 2026 global proxy · market growth = 2025 nowcast/estimates."
                   + (f" Cross-check: Merck's own view of the global Discovery market is '~LSD' mid-term CAGR "
                      f"({cagr['low']:.0%}–{cagr['high']:.0%}, CMD 2025 p.14) – below the modelled market growth."
                      if cagr else ""))
    with right:
        rows = []
        for k, label in br.segments.items():
            w, v, gg, mx = (br.pct(x) for x in (br.w[k], br.v[k], br.g[k], br.draws["mix_s"][k]))
            rows.append({"segment": label, "DS weight w": ui.pct_range(w[0], w[2]),
                         "market weight v": ui.pct_range(v[0], v[2]), "market growth g": ui.pct_range(gg[0], gg[2], 1),
                         "mix contribution": ui.pp_range(mx[0], mx[2])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(ui.badge("C"))
        st.info("**Share effect is the residual** (DS growth − market growth − mix). Splitting it by segment needs "
                "DS growth per customer segment – see *With your data*.", icon=":material/info:")

    with st.expander("How to read this page"):
        st.markdown(
            "- **Grades**: A = disclosed, B = triangulated from 2+ sources, C = estimate. C figures are always ranges "
            "(P10–P90 of a Monte Carlo over all inputs).\n"
            "- **Market methods**: A = Thermo Fisher + Avantor revenue restated to DS-type EMEA Academia, grossed up "
            "for everyone else (others' share from tender win shares); B = academic R&D spend (Eurostat, OECD) x "
            "consumables intensity. A cell is published only if A and B agree within 15%; otherwise the range is the "
            "overlap with two independent checks – Merck's own Discovery market (CMD 2025) scaled down, and a German "
            "bottom-up floor from register filings.\n"
            "- **Sidebar sliders** move the most likely value of the key assumptions; every number recalculates.\n"
            "- Every number traces to its source on *Sources & assumptions*.")


METHOD_LABEL = {"A": "A", "B": "B", "merck_anchor": "the Merck anchor", "de_floor": "the German floor"}


def reconciliation_line(s) -> str:
    """One sentence: which checks the published range is the overlap of, and what was left out (remaining gap)."""
    from src.models.market_model import TOTAL as T

    mk = s.cell(T, "reconciled")
    if mk.status == "published" or not mk.members:
        return ""
    names = [METHOD_LABEL.get(m, m) for m in mk.members]
    out = "The range is the overlap of " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1
                                            else names[0]) + "."
    for m, d in mk.excluded.items():
        label = METHOD_LABEL.get(m, m)
        out += (f" {label[0].upper() + label[1:]} is left out – it sits {ui.eur(abs(d))} "
                f"{'above' if d > 0 else 'below'} the range (remaining gap).")
    return out


def ui_pct(x, q: float) -> float:
    import numpy as np

    return float(np.percentile(x, q))
