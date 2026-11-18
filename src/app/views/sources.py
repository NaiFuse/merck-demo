"""Sources & assumptions: trace every displayed number to its inputs, quotes, files/pages and URLs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.app import ui
from src.models.market_model import TOTAL
from src.models.peer_restatement import Assumption, FactIndex
from src.models.scenario import merck_assumptions

LINK = st.column_config.LinkColumn


def _assumption_df(items, overrides: dict) -> pd.DataFrame:
    rows = [ui.assumption_row(o, a, f"{o}.{a.key}" in overrides) for o, a in items]
    df = pd.DataFrame(rows)
    return df.drop(columns=["low", "base", "high"]) if len(df) else df


def _show_inputs(items, overrides) -> None:
    df = _assumption_df(items, overrides)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"source": st.column_config.TextColumn(width="large"),
                                "quote": st.column_config.TextColumn(width="large")})


def _peer_facts(cfg, facts) -> pd.DataFrame:
    idx = FactIndex(facts, cfg.peer)
    refs = [("1 total revenue", cfg.revenue), ("2 Europe revenue", cfg.geo.numerator),
            ("2 revenue (denominator)", cfg.geo.denominator)]
    refs += [(f"3 in scope ({i.weight.key})", i.ref) for i in cfg.scope.include]
    refs += [(f"3 deduct ({i.weight.key})", i.ref) for i in cfg.scope.deduct]
    rows = []
    for role, ref in refs:
        try:
            f = idx.get(ref)
        except KeyError as e:
            rows.append({"role": role, "value": None, "unit": "", "period": ref.period, "quote": str(e), "source": ""})
            continue
        rows.append({"role": role, "value": f.value, "unit": f.unit, "period": f.period, "grade": f.grade,
                     "quote": f.quote, "source": f.url_or_path, "retrieved": f.retrieved_at})
    return pd.DataFrame(rows)


def trace(ctx, what: str) -> None:
    s, b = ctx.s, ctx.base
    ov = s.overrides
    m = s.model.market
    clusters = [("market", a) for a in m.clusters.values()]
    merck = [("merck_ds", a) for a in merck_assumptions(s, b)]
    d_in = [("market", Assumption.model_validate(x))
            for x in [m.demand["intensity"], m.demand["non_rd_uplift"], m.demand["growth_offset"],
                      *m.demand["sector_weights"].values()]]
    a_in = [("market", Assumption.model_validate(
        m.supply["others_share"]))]

    if what == "DS Academia revenue (3 clusters)":
        q = s.ds.academia_pct(TOTAL)
        st.markdown(f"**{ui.eur_range(q[0], q[2])}** {ui.badge('C', True)}")
        st.latex(r"DS_{Acad,k} = \underbrace{Rev_{DS,2025}\times s_{Europe}}_{\text{DS EMEA revenue (slider)}}"
                 r"\times s_{Academia}\times s_{cluster,k}")
        st.markdown(f"DS EMEA revenue base: **€{s.ds.emea_target:,.0f}m** "
                    f"({'slider' if 'merck_ds.ds_emea_revenue' in ov else 'public-data default'}; "
                    f"default €{s.ds.emea_default:,.0f}m = {b.merck.estimate['ds_revenue_fy2025'].base:,.0f} × "
                    f"{b.merck.estimate['ds_europe_share'].base:.0%}).")
        _show_inputs([x for x in merck if x[1].key in {*b.merck.disclosed, *b.merck.estimate}] + clusters, ov)
        st.caption("Cluster shares: derived by the research base from Eurostat, OECD MSTI, OpenAlex and CORDIS facts "
                   "(see *Facts* tab, sources eurostat / oecd_msti / openalex / cordis).")
    elif what == "DS share of the Academia market":
        sh = s.ds.shares[TOTAL]
        st.markdown(f"**{ui.figure(sh.low, sh.base, sh.high, sh.grade, 'pct')}** {ui.badge(sh.grade, True)}")
        mk = s.cell(TOTAL, "reconciled")
        st.markdown("= DS Academia revenue ÷ market. Per method (draw by draw): "
                    + "; ".join(f"{k}: {ui.pct_range(v[0], v[2], 1)}" for k, v in sh.vs.items())
                    + f". Against the published market range {ui.eur_range(mk.low, mk.high)} ({mk.status}): "
                    "DS P10 ÷ market high .. DS P90 ÷ market low.")
        st.caption(mk.note)
        st.markdown("Inputs: see *DS Academia revenue*, the two methods and the two checks.")
    elif what == "Market – method A (supply side)":
        c = s.cell(TOTAL, "A")
        st.markdown(f"**{ui.eur_range(c.low, c.high)}** {ui.badge(c.grade, True)}")
        st.latex(r"A = \frac{TMO_{restated}+Avantor_{restated}}{1-s_{others}}")
        st.caption("Others' share from tender win shares (see *Others' share (tenders)*).")
        st.markdown(" · ".join(f"{r.label}: {ui.eur_range(r.mc['total'][0], r.mc['total'][2])}"
                               for r in s.peers.results) + " (see the peer traces)")
        _show_inputs(a_in, ov)
    elif what == "Market – method B (demand side)":
        c = s.cell(TOTAL, "B")
        st.markdown(f"**{ui.eur_range(c.low, c.high)}** {ui.badge(c.grade, True)}")
        st.latex(r"B = intensity \times (HES + w_{gov}GOV + w_{pnp}PNP) \times (1+uplift)")
        _show_inputs(d_in, ov)
        st.markdown("**R&D spend facts** (EUR m; GBP converted at the ECB annual average)")
        f = pd.DataFrame([{"country": r.entity, "sector": r.segment, "year": r.period, "value": r.value,
                           "unit": r.unit, "quote": r.quote, "source": r.url_or_path}
                          for r in b.facts if r.metric == "rd_expenditure"])
        sel = st.multiselect("Countries", sorted(f["country"].unique()),
                             default=[c for c in ["DE", "FR", "GB"] if c in set(f["country"])], key="tr_c")
        st.dataframe(f[f["country"].isin(sel)], hide_index=True, width="stretch",
                     column_config={"source": LINK("source")})
    elif what in {r.label + " – restated" for r in s.peers.results}:
        r = next(x for x in s.peers.results if x.label + " – restated" == what)
        cfg = next(c for c in s.peers.configs if c.peer == r.peer)
        q = r.mc["total"]
        st.markdown(f"**{ui.eur_range(q[0], q[2])}** {ui.badge(r.grade, True)} – 6 steps on *Peer comparison*.")
        st.markdown("**Disclosed facts (SEC EDGAR XBRL)** – the quote is the exact XBRL row")
        st.dataframe(_peer_facts(cfg, b.facts), hide_index=True, width="stretch", column_config={"source": LINK("source")})
        st.markdown(f"**Assumptions** (10-K: [{cfg.tenk_url.rsplit('/', 1)[-1]}]({cfg.tenk_url}))")
        _show_inputs([(cfg.peer, a) for a in cfg.assumptions()] + clusters, ov)
        st.caption(f"USD→EUR {r.fx_rate:.4f}: average of 12 ECB monthly reference rates ({r.fx_source}).")
    elif what == "Share bridge":
        br = s.bridge
        g = {k: br.pct(br.draws[k]) for k in ("g_m", "mix", "share", "g_ds")}
        st.markdown(f"DS growth {ui.pct_range(g['g_ds'][0], g['g_ds'][2], 1)} = market {ui.pct_range(g['g_m'][0], g['g_m'][2], 1)}"
                    f" + mix {ui.pp_range(g['mix'][0], g['mix'][2])} + share {ui.pp_range(g['share'][0], g['share'][2])} "
                    + ui.badge("C", True))
        bk = {"ds_emea_organic_growth", "ds_pharma_share_of_non_academia", "market_w_academia",
              "market_pharma_share_of_non_academia", "market_growth_pharma_biotech", "market_growth_industrial_testing",
              "ds_academia_share", "ds_organic_growth_q3_2026"}
        _show_inputs([x for x in merck if x[1].key in bk], ov)
        ga = br.pct(br.g["academia"])
        st.caption(f"Academia market growth {ui.pct_range(ga[0], ga[2], 1)} is derived: method B total {s.year} ÷ "
                   f"{s.year - 1} − 1 per draw (nowcast growth = 2019–23 CAGR + offset).")
    elif what == "Others' share (tenders)":
        from src.models.others_share import views
        o = m.supply["others_share"]
        st.markdown(f"**{ui.pct_range(o['low'], o['high'])}** (median {o['base']:.1%}) {ui.badge(o['grade'], True)}")
        st.markdown("= 1 − (Thermo Fisher + Avantor) share of Academia tender awards, TED + UK FTS, default view, "
                    "2024 .. latest pooled; low / base / high = min / median / max of 8 views.")
        try:
            vs = views(ui.tenders())
            st.dataframe(pd.DataFrame([vars(v) for v in vs]), hide_index=True, width="stretch")
        except Exception as e:  # noqa: BLE001
            ui.db_error(e)
        st.caption(o["source"])
        st.markdown("Award rows and notice links: choose *Tender win shares*.")
    elif what == "Merck market anchor":
        an = s.check_info.get("anchor")
        if not an:
            st.info(s.check_info.get("anchor_error", "not available"))
            return
        c = next(x for x in s.checks[TOTAL] if x.method == "merck_anchor")
        q = c.pct
        st.markdown(f"**{ui.eur_range(q[0], q[2])}** {ui.badge('C', True)}")
        st.latex(r"anchor_k = 80{,}000 \times s_{Europe} \times s_{cluster,k} \times w_{Academia} \times overlap")
        f = an["fact"]
        st.markdown(f"**Disclosed fact** {ui.badge(f.grade, True)}: `{f.entity}` {f.period} {f.metric} = "
                    f"{f.value:g} {f.unit} ({f.region}, {f.segment}) – `{f.url_or_path}`")
        st.code(f.quote, language=None, wrap_lines=True)
        _show_inputs([("derived", an["europe_share"]), ("merck_ds", an["academia_weight"]),
                      ("merck_ds", an["scope_overlap"])] + clusters, ov)
    elif what == "German register floor":
        fl = s.check_info.get("floor")
        if not fl:
            st.info(s.check_info.get("floor_error", "not available"))
            return
        de = np.percentile(fl["de"], (10, 50, 90))
        st.markdown(f"Germany: **{ui.eur_range(de[0], de[2])}** {ui.badge('C', True)} – lower bound")
        st.markdown("= Fisher Scientific GmbH (consumables + chemicals) × Inland share × academic share "
                    "+ VWR International GmbH (chemicals + consumables part of 'Labortechnik') × Deutschland share × "
                    f"academia share + Carl Roth Inland ({fl['carl_roth_period']}, €{fl['carl_roth_inland']:,.1f}m) × "
                    "DS-category share × academic share. Scaled to DACH / 3 clusters by Germany's research share.")
        reg = pd.DataFrame([{"entity": r.entity, "period": r.period, "segment / region": r.segment or r.region,
                             "value": r.value, "unit": r.unit, "quote": r.quote, "file": r.url_or_path}
                            for r in b.facts if r.entity in ("fisher_scientific_gmbh", "vwr_international_gmbh",
                                                             "carl_roth") and r.metric == "revenue"])
        st.dataframe(reg, hide_index=True, width="stretch")
        _show_inputs([("market", a) for a in m.dach_anchor.values()], ov)
    elif what == "Tender win shares":
        from src.models.tender_report import in_scope
        try:
            d = in_scope(ui.tenders())
        except Exception as e:  # noqa: BLE001
            ui.db_error(e)
            return
        st.markdown("Each award row comes from a published notice (TED or UK FTS) – `evidence` names the exact field.")
        grp = st.multiselect("Winner group", sorted(d["group"].unique()), default=["merck"], key="tr_g")
        show = d[d["group"].isin(grp)][["notice_id", "period", "country", "buyer", "title", "winner_name", "value_eur",
                                        "grade", "alignment", "evidence", "is_framework", "source"]]
        urls = _notice_urls()
        show = show.assign(notice_url=show["notice_id"].map(urls))
        st.dataframe(show.sort_values("value_eur", ascending=False), hide_index=True, width="stretch",
                     column_config={"notice_url": LINK("notice"), "value_eur": st.column_config.NumberColumn(format="€%.0f")})


@st.cache_data(ttl=600)
def _notice_urls() -> dict:
    df = ui._read("SELECT notice_id, source_url FROM tenders")
    return dict(zip(df["notice_id"], df["source_url"]))


TRACES = ["DS Academia revenue (3 clusters)", "DS share of the Academia market", "Market – method A (supply side)",
          "Market – method B (demand side)", "Others' share (tenders)", "Merck market anchor", "German register floor",
          "Thermo Fisher – restated", "Avantor – restated", "Share bridge", "Tender win shares"]


def render(ctx) -> None:
    s = ctx.s
    st.title("Sources & assumptions")
    st.caption("Rule 1: no number without provenance. Disclosed inputs carry the exact quote and file/page or URL; "
               "quotes are verified against the source on every app start. Estimates are grade C and shown as ranges.")
    t1, t2, t3, t4 = st.tabs(["Trace a number", "All inputs (live)", "Facts (database)", "Market cells (database)"])
    with t1:
        what = st.selectbox("Figure", TRACES, key="trace_what")
        with st.container(border=True):
            trace(ctx, what)
    with t2:
        items = s.assumptions() + [("merck_ds", a) for a in merck_assumptions(s, ctx.base)]
        df = _assumption_df(items, s.overrides)
        c1, c2, c3 = st.columns(3)
        owners = c1.multiselect("Owner", sorted(df["owner"].unique()), key="as_o")
        grades = c2.multiselect("Grade", ["A", "B", "C"], key="as_g")
        q = c3.text_input("Search", key="as_q")
        if owners:
            df = df[df["owner"].isin(owners)]
        if grades:
            df = df[df["grade"].isin(grades)]
        if q:
            df = df[df.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        st.dataframe(df, hide_index=True, width="stretch", height=520)
        st.caption(f"{len(df)} inputs. 'moved' = changed by a sidebar slider in this session. Same inputs are written "
                   "to the assumptions table by `uv run python -m src.facts_store load`.")
    with t3:
        try:
            f = ui.facts_table()
        except Exception as e:  # noqa: BLE001
            ui.db_error(e)
        else:
            c1, c2, c3 = st.columns(3)
            src = c1.multiselect("Source", sorted(f["source"].unique()), key="f_s")
            ent = c2.multiselect("Entity", sorted(f["entity"].unique()), key="f_e")
            met = c3.multiselect("Metric", sorted(f["metric"].unique()), key="f_m")
            for col, sel in (("source", src), ("entity", ent), ("metric", met)):
                if sel:
                    f = f[f[col].isin(sel)]
            st.dataframe(f, hide_index=True, width="stretch", height=520,
                         column_config={"url_or_path": LINK("source URL / file"),
                                        "quote": st.column_config.TextColumn(width="large")})
            st.caption(f"{len(f):,} facts. Derived facts (source 'derived') record the model and inputs in the quote.")
    with t4:
        try:
            mc = ui.db_market_cells()
        except Exception as e:  # noqa: BLE001
            ui.db_error(e)
        else:
            st.dataframe(mc, hide_index=True, width="stretch")
            st.caption("Last persisted run with public-data defaults (`src.facts_store load`). The pages show the live "
                       "scenario, which equals these values when no slider is moved.")
            live = ctx.default.cell(TOTAL, "reconciled")
            row = mc[(mc["cluster"] == TOTAL) & (mc["method"] == "reconciled") & (mc["year"] == s.year)]
            if len(row):
                ok = np.isclose(row["value_low"].iloc[0], round(live.low, 1)) and \
                     np.isclose(row["value_high"].iloc[0], round(live.high, 1))
                st.markdown((":green[Live default matches the database.]" if ok else
                             ":orange[Live default differs from the database – re-run the loader.]"))
