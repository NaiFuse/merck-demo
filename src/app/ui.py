"""Shared formatting, grade badges, colours and cached data loaders for the Streamlit pages.

Rule 3 lives here: figure() renders every grade-C value as a range, never as a point.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.peer_restatement import Assumption
from src.models.taxonomy import load_taxonomy

TAX = load_taxonomy()
CLUSTERS = {k: v["label"] for k, v in TAX["clusters"].items()}
CLUSTER_LABEL = {**CLUSTERS, "total": "3 clusters total"}
GRADE_COLOR = {"A": "green", "B": "blue", "C": "orange"}
GRADE_TEXT = {"A": "disclosed", "B": "triangulated (2+ sources)", "C": "estimate – shown as range"}

# Categorical slots in fixed order (dataviz reference palette); colour follows the entity, never its rank.
COLOR = {
    "thermo_fisher": "#2a78d6", "Thermo Fisher": "#2a78d6",
    "avantor": "#eb6834", "Avantor": "#eb6834",
    "merck": "#1baf7a", "Merck": "#1baf7a", "Merck DS": "#1baf7a", "merck_ds": "#1baf7a",
    "Multi-brand distributors": "#eda100",
    "Others": "#8a8984",
    "A": "#2a78d6", "B": "#eb6834",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLUSTER_COLOR = dict(zip(CLUSTERS, SERIES[:3]))


# ---------------------------------------------------------------- formatting

def eur(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"€{x / 1000:,.2f}bn" if abs(x) >= 1000 else f"€{x:,.0f}m"


def eur_range(lo: float, hi: float) -> str:
    if abs(hi) >= 1000:
        d = 2 if lo > 0 and hi / lo < 1.6 else 1     # narrow ranges need the extra digit
        return f"€{lo / 1000:,.{d}f}–{hi / 1000:,.{d}f}bn"
    return f"€{lo:,.0f}–{hi:,.0f}m"


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}%}"


def pct_range(lo: float, hi: float, d: int = 0) -> str:
    return f"{lo * 100:.{d}f}–{hi * 100:.{d}f}%"


def pp(x: float) -> str:
    return f"{x * 100:+.1f}pp"


def pp_range(lo: float, hi: float) -> str:
    return f"{lo * 100:+.1f} to {hi * 100:+.1f}pp"


def figure(lo: float | None, base: float | None, hi: float | None, grade: str, kind: str = "eur") -> str:
    """Display string honouring rule 3: grade C (or no base) -> range only; A/B -> base (P10–P90)."""
    rng = {"eur": eur_range, "pct": pct_range, "pp": pp_range}[kind]
    one = {"eur": eur, "pct": pct, "pp": pp}[kind]
    if grade == "C" or base is None:
        return rng(lo, hi)
    if lo is None or hi is None or lo == hi:
        return one(base)
    return f"{one(base)} ({rng(lo, hi)})"


def badge(grade: str, short: bool = False) -> str:
    text = f"Grade {grade}" if short else f"Grade {grade} · {GRADE_TEXT[grade]}"
    return f":{GRADE_COLOR[grade]}-badge[{text}]"


def triple_str(a: Assumption) -> str:
    if a.unit in ("share", "%", "share points"):
        f = (lambda v: f"{v:.1%}") if a.unit != "%" else (lambda v: f"{v:g}%")
    elif "_" in a.unit:
        f = lambda v: f"{v:,.0f}"  # noqa: E731
    else:
        f = lambda v: f"{v:g}"  # noqa: E731
    if a.low == a.high:
        return f(a.base)
    if a.grade == "C":
        return f"{f(a.low)} – {f(a.high)}"
    return f"{f(a.base)} ({f(a.low)} – {f(a.high)})"


def assumption_row(owner: str, a: Assumption, overridden: bool = False) -> dict:
    src = a.source
    if a.source_file:
        src_ref = f"{a.source_file} p.{a.page}"
    elif a.source.startswith("derived:"):
        src_ref = "derived (research base)"
    else:
        src_ref = ""
    return {"owner": owner, "key": a.key, "value": triple_str(a), "low": a.low, "base": a.base, "high": a.high,
            "unit": a.unit, "grade": a.grade, "slider": "moved" if overridden else "", "source": src,
            "file / page": src_ref, "quote": a.quote or a.evidence or "", "note": a.note or ""}


# ---------------------------------------------------------------- charts

def style(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def range_chart(rows: list[dict], x_title: str, height: int | None = None, fmt=eur) -> go.Figure:
    """Horizontal P10–P90 range bars. rows: {y, series, p10, p50, p90, color}. P50 shown only on hover (rule 3)."""
    fig = go.Figure()
    seen = set()
    ys = list(dict.fromkeys(r["y"] for r in rows))
    series = list(dict.fromkeys(r["series"] for r in rows))
    offs = {s: (i - (len(series) - 1) / 2) * 0.22 for i, s in enumerate(series)}
    for r in rows:
        yi = ys.index(r["y"]) + offs[r["series"]]
        show = r["series"] not in seen
        seen.add(r["series"])
        fig.add_trace(go.Scatter(
            x=[r["p10"], r["p90"]], y=[yi, yi], mode="lines+markers", name=r["series"], legendgroup=r["series"],
            showlegend=show, line=dict(color=r["color"], width=10), marker=dict(size=10, color=r["color"]),
            hovertemplate=(f"<b>{r['series']}</b> · {r['y']}<br>P10–P90 {fmt(r['p10'])} – {fmt(r['p90'])}"
                           f"<br>P50 {fmt(r['p50'])}<extra></extra>")))
    fig.update_yaxes(tickvals=list(range(len(ys))), ticktext=ys, autorange="reversed", showgrid=False)
    fig.update_xaxes(title=x_title, showgrid=True, gridcolor="rgba(128,128,128,0.18)", rangemode="tozero")
    return style(fig, height or max(220, 70 * len(ys) + 90))


# ---------------------------------------------------------------- data (read-only DB, closed immediately)

def db_error(e: Exception) -> None:
    if isinstance(e, FileNotFoundError):
        st.warning("No database yet. Create it with `uv run python -m src.db` and load it.")
    elif isinstance(e, (duckdb.IOException, duckdb.ConnectionException)):
        st.info("Database is busy (a loader is writing). Refresh in a moment.")
    else:
        st.error(f"Database read failed: {e}")


def _read(sql: str) -> pd.DataFrame:
    from src.db import connect_readonly

    with connect_readonly() as con:
        return con.execute(sql).df()


@st.cache_data(ttl=600, show_spinner="Loading tender awards…")
def tenders() -> pd.DataFrame:
    from src.db import connect_readonly
    from src.models.tender_report import load_awards

    with connect_readonly() as con:
        return load_awards(con)


@st.cache_data(ttl=600)
def facts_table() -> pd.DataFrame:
    return _read("""
        SELECT f.id, f.entity, f.period, f.metric, f.value, f.unit, f.region, f.segment, f.grade, f.quote,
               d.source, d.url_or_path, d.retrieved_at
        FROM facts f JOIN documents d ON d.id = f.document_id ORDER BY f.entity, f.metric, f.period""")


@st.cache_data(ttl=600)
def db_assumptions() -> pd.DataFrame:
    return _read("SELECT key, value_low, value_base, value_high, grade, source, note FROM assumptions ORDER BY key")


@st.cache_data(ttl=600)
def db_market_cells() -> pd.DataFrame:
    return _read("SELECT * FROM market_cells ORDER BY method, cluster, year")
