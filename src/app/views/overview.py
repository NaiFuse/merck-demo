"""Overview: three headline findings (src/models/findings.py) around a compact market strip.

Panel 1 (lead): German register data side by side – Fisher Scientific GmbH academic growth, VWR decline, DS growth.
Market strip: DS Academia revenue, market range (A–B overlap) with the Merck anchor band, DS share.
Panel 2: DS's catalogue model runs below the tender radar.  Panel 3 (closing): dual-view share bridge.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app import ui
from src.models.merck_ds import TOTAL

DISCLAIMER = ("**Unofficial demo** built from free public data only – not endorsed by or affiliated with Merck KGaA, "
              "Darmstadt, Germany. Grades: A disclosed · B triangulated · C estimate (always a range).")
GROUP_COLOR = {"thermo_fisher": ui.COLOR["thermo_fisher"], "avantor": ui.COLOR["avantor"], "merck": ui.COLOR["merck"],
               "market": "#8a8984", "other": "#6250d6", "eu": "#8a8984"}
MARKET_SUB = {"model": "2025 nowcast", "merck_lsd": "Merck ~LSD, global"}


@st.cache_data(ttl=600, show_spinner=False)
def _findings(_ctx, overrides: tuple) -> list:
    """Cached per scenario (overrides); _ctx is not hashed."""
    from src.models import findings

    return findings.build(_ctx.s, _ctx.base, ui.tenders())


def get_findings(ctx) -> list | None:
    try:
        return _findings(ctx, tuple(sorted(ctx.s.overrides.items())))
    except Exception as e:  # noqa: BLE001
        ui.db_error(e)
        return None


# ---------------------------------------------------------------- panel helpers

def caveat_box(f, title: str = "How to read this") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption("\n".join(f"- {c}" for c in f.caveats))


def sources_expander(f) -> None:
    with st.expander("Figures and sources"):
        rows = [{"figure": x.label, "value": x.text, "grade": x.grade, "period": x.period,
                 "source": "; ".join(s.ref for s in x.sources), "quote": "; ".join(s.quote for s in x.sources)}
                for x in f.figures]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"quote": st.column_config.TextColumn(width="large")})
        st.caption(f.grade_note)


def growth_chart(f) -> go.Figure:
    figs = [x for x in f.figures if x.value is not None][::-1]
    fig = go.Figure(go.Bar(
        x=[x.value for x in figs], y=[x.label for x in figs], orientation="h",
        marker=dict(color=[GROUP_COLOR.get(x.group, "#8a8984") for x in figs], cornerradius=4),
        text=[x.text for x in figs], cliponaxis=False, textposition="outside", textfont=dict(size=12),
        customdata=[[x.grade, x.sources[0].ref if x.sources else ""] for x in figs],
        hovertemplate="<b>%{y}</b><br>%{text} · grade %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.7)")
    lo, hi = min(0.0, min(x.value for x in figs)), max(x.value for x in figs)
    fig.update_xaxes(tickformat="+.0%", title="growth vs. prior year", showgrid=True, gridcolor="rgba(128,128,128,0.18)",
                     range=[lo - 0.065, hi + 0.04])     # room for outside labels on both sides
    fig.update_yaxes(showgrid=False)
    return ui.style(fig, 34 * len(figs) + 90, showlegend=False, uniformtext=dict(minsize=11, mode="show"))


def bridge_chart(br, view: str = "model", y_range: tuple[float, float] | None = None) -> go.Figure:
    d = br.views.get(view, br.draws)
    q = {k: br.pct(d[k]) for k in ("g_m", "mix", "share", "g_ds")}
    m = {k: v[1] for k, v in q.items()}
    steps = [(f"Market growth<br><sub>{MARKET_SUB[view]}</sub>", 0.0, m["g_m"], q["g_m"], ui.COLOR["Others"]),
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
    fig.update_yaxes(title="growth, %", tickformat=".1%", range=y_range)
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    return ui.style(fig, 360, bargap=0.35, title=dict(text=br.view_labels.get(view, view), font=dict(size=14)))


def reconciliation_line(s) -> str:
    """Range basis, floor check and the Merck anchor band with its gap to the range (EUR and %)."""
    from src.models.market_model import ANCHOR_LABEL, TOTAL as T

    mk = s.cell(T, "reconciled")
    out = "" if mk.status == "published" else (
        "Range = overlap of A and B." if mk.status == "range" else "A and B do not overlap; range = min–max.")
    if mk.floor_check:
        out += (" German register floor below it (consistent)." if mk.floor_check == "ok"
                else " :orange[German register floor above the range low – check.]")
    band = mk.bands.get("merck_anchor")
    if band:
        where = "above" if band["gap_eur"] > 0 else "below" if band["gap_eur"] < 0 else "overlapping"
        out += (f" Shown separately – *{ANCHOR_LABEL}*: {ui.eur_range(band['low'], band['high'])}, "
                + (f"{ui.eur(abs(band['gap_eur']))} ({abs(band['gap_pct']):.0%}) {where} our range"
                   if band["gap_eur"] else "overlapping our range") + ".")
    return out


# ---------------------------------------------------------------- page

def panel_germany(f) -> None:
    with st.container(border=True):
        st.markdown("##### 1 · Lead finding · Germany")
        st.subheader(f.title)
        st.markdown(f.lead)
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(growth_chart(f), width="stretch", key="germany_growth")
            st.markdown(ui.badge("A") + " :gray[blue = Thermo Fisher's German distributor · orange = Avantor/VWR · "
                                        "aqua = Merck DS · grey = German R&D spend · violet = Carl Roth]")
        with right:
            caveat_box(f)
        sources_expander(f)


def market_strip(ctx) -> None:
    s = ctx.s
    ds = s.ds.academia_pct(TOTAL)
    mk = s.cell(TOTAL, "reconciled")
    sh = s.ds.shares[TOTAL]
    with st.container(border=True):
        st.markdown("##### Market at a glance · Academia, 3 clusters, FY2025")
        c = st.columns(3)
        c[0].metric("DS Academia revenue (estimate)", ui.eur_range(ds[0], ds[2]), border=True,
                    help="DS EMEA revenue x Academia share x cluster share (P10–P90).")
        c[0].markdown(ui.badge(s.ds.grade, short=True))
        c[1].metric("DS-type Academia market", ui.figure(mk.low, mk.base, mk.high, mk.grade), border=True,
                    help="Published as a point only if supply side (A) and demand side (B) agree within 15%; "
                         "otherwise the overlap of A and B, checked against the German register floor.")
        c[1].markdown(ui.badge(mk.grade, short=True))
        c[2].metric("DS share of that market", ui.figure(sh.low, sh.base, sh.high, sh.grade, "pct"), border=True)
        c[2].markdown(ui.badge(sh.grade, short=True))
        a, b = s.cell(TOTAL, "A"), s.cell(TOTAL, "B")
        st.caption(f"Supply side (A) {ui.eur_range(a.low, a.high)} · demand side (B) {ui.eur_range(b.low, b.high)} · "
                   f"A/B gap {mk.gap_pct:.0%} at the P50. {reconciliation_line(s)} Details: *Academia* page.")


def panel_tenders(f) -> None:
    by = {x.label: x for x in f.figures}

    def pick(prefix):
        return next(x for k, x in by.items() if k.startswith(prefix))

    with st.container(border=True):
        st.markdown("##### 2 · Channel")
        st.subheader(f.title)
        st.markdown(f.lead)
        tiles = [("DS customers spending < €100k a year", pick("DS customers")),
                 ("EU supply threshold, central gov. (2026)", pick("EU threshold, supply contracts – central")),
                 ("Merck direct share of Academia tender value", pick("Merck direct win share of")),
                 ("DS estimated share of the Academia market", pick("DS estimated share"))]
        cols = st.columns(4)
        for c, (label, x) in zip(cols, tiles):
            c.metric(label, x.text, border=True, help=x.label)
            c.markdown(ui.badge(x.grade, short=True))
        left, right = st.columns([3, 2])
        with left:
            rows = []
            for name, key, color in (("Thermo Fisher", "Thermo Fisher win share", ui.COLOR["thermo_fisher"]),
                                     ("Avantor", "Avantor win share", ui.COLOR["avantor"]),
                                     ("Merck (direct)", "Merck direct win share of", ui.COLOR["merck"])):
                lo, hi = (float(v) / 100 for v in pick(key).text.rstrip("%").split("–"))
                rows.append({"y": "Tender win share (value)", "series": name, "p10": lo, "p50": (lo + hi) / 2,
                             "p90": hi, "color": color})
            lo, hi = (float(v) / 100 for v in pick("DS estimated share").text.rstrip("%").split("–"))
            rows.append({"y": "Market share (estimate)", "series": "Merck DS", "p10": lo, "p50": (lo + hi) / 2,
                         "p90": hi, "color": ui.COLOR["merck"]})
            fig = ui.range_chart(rows, "share, % (range)", fmt=lambda v: f"{v:.1%}")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch", key="tender_ranges")
            st.markdown(ui.badge("C") + " :gray[win shares 2024–26 pooled; range = all awards vs. awards with a "
                                        "disclosed per-winner value]")
        with right:
            caveat_box(f)
        sources_expander(f)


def panel_bridge(ctx, f) -> None:
    br = ctx.s.bridge
    with st.container(border=True):
        st.markdown("##### 3 · Growth")
        st.subheader(f.title)
        st.markdown(f.lead)
        st.warning("**Not like-for-like.** DS growth = Q3 2026 vs. Q3 2025, *global* DS (+1.7% organic, disclosed) as "
                   "EMEA proxy. Market growth = 2025 vs. 2024 – left: our nowcast (Academia = method B), right: "
                   "Merck's own '~LSD' view of the global Discovery market.", icon=":material/schedule:")
        views = list(br.views) or ["model"]
        vals = [x for v in views for k in ("g_m", "g_ds") for x in (0.0, br.pct(br.views[v][k])[1])]
        vals += [br.pct(br.views[v]["g_m"])[1] + br.pct(br.views[v]["mix"])[1] for v in views]
        y_range = (min(vals + [0]) - 0.012, max(vals) + 0.012)
        cols = st.columns(len(views))
        for c, v in zip(cols, views):
            with c:
                st.plotly_chart(bridge_chart(br, v, y_range), width="stretch", key=f"bridge_{v}")
        rows = []
        for v in views:
            d = br.views[v]
            q = {k: br.pct(d[k]) for k in ("g_m", "mix", "share", "g_ds")}
            rows.append({"market-growth view": br.view_labels.get(v, v),
                         "market growth": ui.pct_range(q["g_m"][0], q["g_m"][2], 1),
                         "mix effect": ui.pp_range(q["mix"][0], q["mix"][2]),
                         "share effect": ui.pp_range(q["share"][0], q["share"][2]),
                         "DS growth": ui.pct_range(q["g_ds"][0], q["g_ds"][2], 1),
                         "draws with share effect < 0": f"{float((d['share'] < 0).mean()):.0%}"})
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.markdown(ui.badge("C") + " :gray[bars at the P50; labels P10–P90; same DS growth and weights in both]")
            with st.expander("Segment inputs and mix (our-model view)"):
                seg = []
                for k, label in br.segments.items():
                    w, v, gg, mx = (br.pct(x) for x in (br.w[k], br.v[k], br.g[k], br.draws["mix_s"][k]))
                    seg.append({"segment": label, "DS weight w": ui.pct_range(w[0], w[2]),
                                "market weight v": ui.pct_range(v[0], v[2]),
                                "market growth g": ui.pct_range(gg[0], gg[2], 1),
                                "mix contribution": ui.pp_range(mx[0], mx[2])})
                st.dataframe(pd.DataFrame(seg), hide_index=True, width="stretch")
                st.latex(r"g_{DS}-g_M=\underbrace{\textstyle\sum_s (w_s-v_s)\,g_s}_{\text{mix}}"
                         r"+\underbrace{\textstyle\sum_s w_s\,(g_{DS,s}-g_s)}_{\text{share}}")
        with right:
            caveat_box(f)


def render(ctx) -> None:
    st.title("Merck DS in EMEA Academia – outside-in view")
    st.info(DISCLAIMER, icon=":material/info:")
    st.caption("DACH · France & Benelux · UK, Ireland & Nordics · FY2025 · every number traces to its source on "
               "*Sources & assumptions*")
    if ctx.s.overrides:
        st.caption(f":orange[Scenario: {len(ctx.s.overrides)} input(s) moved in the sidebar – model figures differ "
                   "from the public-data default.]")
    fs = get_findings(ctx)
    if not fs:
        st.stop()
    f = {x.key: x for x in fs}
    panel_germany(f["germany_channel"])
    market_strip(ctx)
    panel_tenders(f["tender_radar"])
    panel_bridge(ctx, f["growth_bridge"])
