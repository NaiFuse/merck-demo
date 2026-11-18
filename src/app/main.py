"""Streamlit entry point: uv run streamlit run src/app/main.py"""

import duckdb
import pandas as pd
import streamlit as st

from src.db import connect_readonly, table_counts
from src.models.taxonomy import load_taxonomy
from src.settings import get_settings

st.set_page_config(page_title="Merck DS EMEA – Market Research Demo", layout="wide")
st.title("Merck DS EMEA – Academia market demo")
st.caption("Unofficial outside-in demo built from public data. Not endorsed by Merck.")

st.success("Demo skeleton OK")

settings = get_settings()
st.subheader("Rows per table")
try:
    # Read-only and closed immediately, so loader scripts can take the write lock between page loads.
    with connect_readonly() as con:
        counts = table_counts(con)
except FileNotFoundError:
    st.warning("No database yet. Create it with: `uv run python -m src.db`")
except (duckdb.IOException, duckdb.ConnectionException):
    st.info("Database is busy (a loader is writing). Refresh in a moment.")
except duckdb.CatalogException:
    st.warning("Database schema is incomplete. Run: `uv run python -m src.db`")
else:
    st.dataframe(pd.DataFrame(counts.items(), columns=["table", "rows"]), hide_index=True)
st.caption(f"Database: {settings.duckdb_path}")

with st.expander("Taxonomy"):
    tax = load_taxonomy()
    st.write("Clusters:", {k: v["label"] for k, v in tax["clusters"].items()})
    st.write("Categories:", [v["label"] for v in tax["categories"].values()])

with st.expander("Configured settings (values hidden)"):
    st.write(settings.configured())
