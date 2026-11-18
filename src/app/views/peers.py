"""Peer comparison: restated Thermo Fisher and Avantor vs. the Merck DS estimate, with confidence grades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app import ui
from src.models.market_model import TOTAL
from src.models.scenario import merck_assumptions


def _f(t, unit: str) -> str:
    if unit == "share":
        return f"{t[0]:.1%} / {t[1]:.1%} / {t[2]:.1%}"
    return f"{t[0]:,.0f} / {t[1]:,.0f} / {t[2]:,.0f}"


def tornado(r, top: int = 8) -> go.Figure:
    sens = r.sensitivity[:top][::-1]
    fig = go.Figure(go.Bar(
        x=[abs(x.swing) for x in sens], y=[x.key.split(".", 1)[1] for x in sens], orientation="h",
        marker=dict(color=ui.COLOR[r.peer], cornerradius=4),
        customdata=[[x.swing, x.var_share, x.label] for x in sens],
        hovertemplate="<b>%{y}</b><br>swing %{customdata[0]:+,.0f} EUR m (input low → high)"
                      "<br>variance share %{customdata[1]:.0%}<br>%{customdata[2]}<extra></extra>"))
    fig.update_xaxes(title="swing of 3-cluster total, EUR m", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=False)
    return ui.style(fig, 40 * len(sens) + 90)


def render(ctx) -> None:
    s = ctx.s
    st.title("Peer comparison")
    st.caption(f"DS-comparable EMEA Academia revenue FY{s.year}, EUR m. Peers restated from SEC filings in 6 steps; "
               "Merck DS from public disclosures plus estimates. Output grade = worst input grade.")

    cols = st.columns(len(s.peers.results) + 1)
    for c, r in zip(cols, s.peers.results):
        q = r.mc["total"]
        c.metric(r.label, ui.eur_range(q[0], q[2]), border=True, help="P10–P90, sum of the three clusters")
        c.markdown(ui.badge(r.grade, short=True))
    q = s.ds.academia_pct(TOTAL)
    cols[-1].metric("Merck DS (est.)", ui.eur_range(q[0], q[2]), border=True)
    cols[-1].markdown(ui.badge(s.ds.grade, short=True))

    rows = []
    for r in s.peers.results:
        row = {"company": r.label, "grade": r.grade}
        for k in list(ui.CLUSTERS) + [TOTAL]:
            p = r.mc[k]
            row[ui.CLUSTER_LABEL[k]] = ui.eur_range(p[0], p[2])
        row["min–max (bounding)"] = ui.eur_range(r.total[0], r.total[2])
        mq = r.mc["merck_total"]
        row["of which resold Merck"] = ui.eur_range(mq[0], mq[2])
        rows.append(row)
    row = {"company": "Merck DS (est.)", "grade": s.ds.grade}
    for k in list(ui.CLUSTERS) + [TOTAL]:
        p = s.ds.academia_pct(k)
        row[ui.CLUSTER_LABEL[k]] = ui.eur_range(p[0], p[2])
    row["min–max (bounding)"], row["of which resold Merck"] = "–", "–"
    rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"grade": st.column_config.TextColumn("grade", help="A disclosed · B triangulated · "
                                                                     "C estimate (range)")})
    st.caption("P10–P90 of 10,000 Monte Carlo draws (default display); min–max multiplies all lows / all highs. "
               "Peer revenue keeps resold Merck product, so part of it is Merck volume sold through the channel.")

    st.subheader("Restatement, step by step")
    tabs = st.tabs([r.label for r in s.peers.results] + ["Merck DS (est.)"])
    for tab, r, cfg in zip(tabs, s.peers.results, s.peers.configs):
        with tab:
            steps = pd.DataFrame([{
                "step": f"{x.n} {x.label}", "factor min / base / max": _f(x.factor, "share") if x.factor else "",
                "value min / base / max": _f(x.value, "eur"), "unit": "EUR m" if x.n == "6" else f"{cfg.currency} m",
                "grade": x.grade, "detail": x.detail} for x in r.steps])
            st.dataframe(steps, hide_index=True, width="stretch")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**What drives the range** (one-at-a-time swing)")
                st.plotly_chart(tornado(r), width="stretch")
            with c2:
                st.markdown("**Inputs and grades**")
                moved = {u.split(".", 1)[1] for u in s.overrides if u.startswith(cfg.peer + ".")}
                inp = pd.DataFrame([{"input": a.key, "value": ui.triple_str(a), "grade": a.grade,
                                     "slider": "moved" if a.key in moved else "", "source": a.source}
                                    for a in cfg.assumptions()])
                st.dataframe(inp, hide_index=True, width="stretch", height=360)
            st.caption(f"Revenue facts: SEC EDGAR XBRL ({cfg.tenk_url}). USD→EUR {r.fx_rate:.4f} = FY average of ECB "
                       f"monthly rates.")
    with tabs[-1]:
        m = ctx.base.merck
        d = s.ds
        st.markdown("DS EMEA revenue = DS revenue FY2025 × DS Europe share (sidebar slider sets the base) → × Academia "
                    "share → × cluster share (research base).")
        moved = {u.split(".", 1)[1] for u in s.overrides if u.startswith("merck_ds.")}
        inp = pd.DataFrame([{"input": a.key, "value": ui.triple_str(a), "grade": a.grade,
                             "slider": "moved" if a.key in moved else "", "source": a.source}
                            for a in merck_assumptions(s, ctx.base) if a.key in {*m.disclosed, *m.estimate}])
        st.dataframe(inp, hide_index=True, width="stretch")
        q = [float(x) for x in np.percentile(d.emea, (10, 50, 90))]
        st.markdown(f"DS EMEA revenue (all customers): **{ui.eur_range(q[0], q[2])}** "
                    f"{'(slider)' if 'merck_ds.ds_emea_revenue' in s.overrides else '(public-data default)'} "
                    + ui.badge("C", short=True))

    st.subheader("DACH sanity check against German register filings")
    if not s.peers.crosscheck:
        st.info("Register facts not extracted yet (`uv run python -m src.extract.pdf run`).")
    for c in s.peers.crosscheck:
        with st.container(border=True):
            icon = ":material/check_circle:" if c.plausible else ":material/warning:"
            st.markdown(f"{icon} **{c.peer.replace('_', ' ').title()} vs. {c.anchor_entity}** – {c.verdict}")
            st.markdown(f"Estimate Germany-equivalent {ui.eur_range(c.estimate_de[0], c.estimate_de[2])} vs. register "
                        f"anchor {ui.eur_range(c.anchor[0], c.anchor[2])} (P10–P90). Rule: {c.rule}.")
            st.caption(f"Anchor = {c.inputs}")
            with st.expander("Scope differences"):
                for x in c.scope_differences:
                    st.markdown(f"- {x}")

    with st.expander("Confidence grades"):
        for g in "ABC":
            st.markdown(f"{ui.badge(g)}")
        st.caption("C-grade figures are always shown as ranges. A derived figure takes the worst grade of its inputs.")
