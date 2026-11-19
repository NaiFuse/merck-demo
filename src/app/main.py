"""Streamlit entry point: uv run streamlit run src/app/main.py

Five pages share one sidebar of sliders. Sources are verified once per session (scenario.prepare, cached); every
slider move reruns the Monte Carlo (scenario.run, ~0.05 s). The DB is opened read-only and closed immediately,
so loader scripts can take the write lock between page loads (rule 7).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# `streamlit run` puts src/app on sys.path, not the project root; don't rely on the editable install's .pth file
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import scenario as sc  # noqa: E402

st.set_page_config(page_title="Merck DS EMEA – Academia demo", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Verifying sources and loading caches (first load only, ~15 s)…")
def base() -> sc.Base:
    return sc.prepare()


@st.cache_resource(max_entries=64, show_spinner=False)
def scenario(overrides: tuple[tuple[str, float], ...]) -> sc.Scenario:
    return sc.run(base(), dict(overrides))


@dataclass
class Ctx:
    base: sc.Base
    s: sc.Scenario              # current scenario (sliders applied)
    default: sc.Scenario        # public-data default


def _reset() -> None:
    for k in sc.KNOBS:
        st.session_state[k.uid] = _ui_default(k)


def _ui_default(k: sc.Knob) -> float:
    v = base().default(k.uid)
    return round(v * 100, 2) if k.pct else round(v, 0)


def sidebar() -> dict[str, float]:
    st.sidebar.markdown("### Scenario")
    st.sidebar.caption("Sliders move the most likely value of an input; its uncertainty width is kept. "
                       "Everything recalculates live. Moved inputs stay grade C.")
    out = {}
    for group in dict.fromkeys(k.group for k in sc.KNOBS):
        with st.sidebar.expander(group, expanded=group == "Merck DS"):
            for k in (k for k in sc.KNOBS if k.group == group):
                d = _ui_default(k)
                st.session_state.setdefault(k.uid, d)
                if k.pct:
                    v = st.slider(k.label, k.lo * 100, k.hi * 100, key=k.uid, step=k.step * 100, format="%.1f%%",
                                  help=(k.help + " " if k.help else "") + f"Public-data default: {d:.1f}%")
                    out[k.uid] = v / 100
                else:
                    v = st.slider(k.label, float(k.lo), float(k.hi), key=k.uid, step=float(k.step), format="€%.0fm",
                                  help=(k.help + " " if k.help else "") + f"Public-data default: €{d:,.0f}m")
                    out[k.uid] = v
    moved = [k for k in sc.KNOBS if abs(st.session_state[k.uid] - _ui_default(k)) > 1e-9]
    if moved:
        st.sidebar.warning(f"{len(moved)} input(s) moved from the public-data default.")
    st.sidebar.button("Reset to public-data defaults", on_click=_reset, disabled=not moved, width="stretch")
    st.sidebar.divider()
    st.sidebar.caption("Unofficial outside-in demo from free public data. Not endorsed by Merck KGaA.")
    return {k.uid: out[k.uid] for k in moved}


def main() -> None:
    try:
        b = base()
    except Exception as e:  # noqa: BLE001 – show the reason instead of a stack trace
        st.error(f"Could not prepare the model: {e}")
        st.stop()
    overrides = sidebar()
    ctx = Ctx(b, scenario(tuple(sorted(overrides.items()))), scenario(()))
    from src.app.report_button import report_button

    report_button(ctx, where="sidebar")

    from src.app.views import academia, overview, peers, phase2, sources

    pages = [
        st.Page(lambda: overview.render(ctx), title="Overview", icon=":material/insights:", url_path="overview",
                default=True),
        st.Page(lambda: academia.render(ctx), title="Academia", icon=":material/school:", url_path="academia"),
        st.Page(lambda: peers.render(ctx), title="Peer comparison", icon=":material/compare_arrows:", url_path="peers"),
        st.Page(lambda: sources.render(ctx), title="Sources & assumptions", icon=":material/fact_check:",
                url_path="sources"),
        st.Page(lambda: phase2.render(ctx), title="With your data", icon=":material/lock:", url_path="with-your-data"),
    ]
    st.navigation(pages).run()


main()
