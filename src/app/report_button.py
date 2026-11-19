"""One-click PDF export of the Academia report (src/report/academia_report.py) for the current scenario."""

from __future__ import annotations

from datetime import date

import streamlit as st

from src.app import ui


@st.cache_data(ttl=600, show_spinner="Building the PDF report…", max_entries=16)
def _pdf(_ctx, overrides: tuple) -> bytes:
    """Cached per scenario (overrides); _ctx is not hashed."""
    from src.report.academia_report import build

    return build(_ctx.s, _ctx.base, ui.tenders())


def report_button(ctx, where: str = "sidebar") -> None:
    try:
        data = _pdf(ctx, tuple(sorted(ctx.s.overrides.items())))
    except Exception as e:  # noqa: BLE001
        (st.sidebar if where == "sidebar" else st).error(f"Report could not be built: {e}")
        return
    target = st.sidebar if where == "sidebar" else st
    target.download_button("Download Academia report (PDF)", data, file_name=f"merck-ds-academia-report_{date.today()}.pdf",
                           mime="application/pdf", icon=":material/picture_as_pdf:", key=f"report_{where}",
                           width="stretch", help="Findings, market range with the Merck anchor band, peers, tender win "
                                                 "shares, sources and the 'With your data' page – current scenario.")
