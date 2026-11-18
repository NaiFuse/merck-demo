"""With your data: phase-2 modules, shown greyed out (inactive) – what Merck's own data would unlock."""

from __future__ import annotations

import streamlit as st

GREY = """
<style>
.phase2 {opacity: 0.55; filter: grayscale(1);}
</style>
"""

MODULES = [
    ("Exact DS share", ":material/target:",
     "Upload DS sales by country × category × customer segment. Replaces the public-data estimate (grade C range) "
     "with your actuals (grade A), so the DS share narrows to the market range alone.",
     "DS sales by country / category / segment (CSV)"),
    ("Share bridge by segment", ":material/stacked_bar_chart:",
     "DS growth per customer segment splits the share effect by segment (Academia, Pharma & Biotech, Industrial & "
     "Testing) instead of showing it as a residual.",
     "DS organic growth by segment and cluster"),
    ("Pharma segment", ":material/medication:",
     "Same two-method market model for Pharma & Biotech: supply side from restated peers, demand side from BERD "
     "(business R&D) in pharma (NACE C21) and biotech.",
     "Segment tagging of DS customers"),
    ("Industry & Service segment", ":material/factory:",
     "Industrial & testing labs: demand side from lab-intensive industries (chemicals, food, environmental testing).",
     "Segment tagging of DS customers"),
    ("Customer survey", ":material/forum:",
     "Share-of-wallet and switching drivers from a short survey of lab managers and purchasers in the three clusters.",
     "Survey responses (CSV)"),
    ("Category split", ":material/category:",
     "Market and share per DS category (chemistry catalogue, solvents, analytical reagents, biology catalogue, lab "
     "consumables) – today all categories are combined.",
     "DS sales by category"),
]


def render(ctx) -> None:
    st.title("With your data")
    st.caption("Phase 2 – modules that need Merck's internal data. Shown inactive: nothing is uploaded or stored.")
    st.info("Everything in this demo comes from free public data. With your sales data, the grade-C ranges for DS "
            "become grade-A figures, and the remaining uncertainty is the market itself.", icon=":material/lock:")
    st.markdown(GREY, unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (title, icon, text, needs) in enumerate(MODULES):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f'<div class="phase2"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
                st.markdown(f"{icon} :gray[Needs: {needs}] :gray-badge[Phase 2]")
                st.file_uploader(f"Upload – {title}", disabled=True, key=f"p2_{i}", label_visibility="collapsed")

    with st.container(border=True):
        st.markdown('<div class="phase2"><h4>Preview: DS share with exact revenue</h4></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.number_input("DS Academia revenue, 3 clusters (EUR m, your actuals)", value=0.0, disabled=True, key="p2_rev")
        c2.selectbox("Segment", ["Academia", "Pharma & Biotech", "Industrial & Testing"], disabled=True, key="p2_seg")
        st.caption("Inactive in this demo. Until then: the *DS EMEA revenue* slider in the sidebar moves the estimate.")
