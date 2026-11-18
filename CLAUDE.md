# Merck DS EMEA – Market Research Demo

## Purpose
Outside-in pitch demo: estimate Merck Discovery Solutions' (DS) position in the EMEA Academia
segment vs. Thermo Fisher and Avantor, using only free public data. Unofficial, not endorsed by Merck.

## Non-negotiable rules
1. No number without provenance: every fact stores source_url (or file + page), retrieval date and
   the exact quote/row it came from. Code verifies the quote contains the number.
2. LLM only reads unstructured text (PDFs, filings). API data goes straight to the DB. All
   arithmetic is plain Python – never ask the LLM to calculate.
3. Every derived figure has a confidence grade: A = disclosed, B = triangulated from 2+ sources,
   C = estimate. C-grade figures are always shown as ranges.
4. Official APIs only. Respect terms of use and rate limits. No scraping of supplier web shops.
5. Secrets only in .env. Never print or commit keys.
6. Small, testable modules. After each milestone: run it, show sample output, list open issues.

## Stack
Python 3.12, uv, DuckDB (data/demo.duckdb), httpx, pydantic, pandas, Anthropic SDK
(model name from env var ANTHROPIC_MODEL), Streamlit + Plotly for the dashboard, pytest.

## Folder structure
src/connectors/   one file per source (ted, edgar, eurostat, companies_house, inpi, cordis, openalex)
src/extract/      LLM extraction from PDFs with quotes
src/models/       taxonomy, peer restatement, demand model, share bridge
src/app/          Streamlit pages
config/           taxonomy.yaml, assumptions/*.yaml (one per peer + market drivers)
data/raw/         manually downloaded PDFs (merck, peers, registers)
tests/

## Database tables
documents(id, source, url_or_path, retrieved_at)
facts(id, document_id, entity, period, metric, value, unit, region, segment, quote, grade)
assumptions(id, key, value_low, value_base, value_high, source, grade, note)
tenders(notice_id, country, buyer, cpv, value_eur, award_date, winner_name, winner_id)
market_cells(segment, category, cluster, year, method, value_low, value_base, value_high, grade)

## Demo scope
Segment: Academia (universities, research institutes, hospitals, government labs).
Peers: Thermo Fisher, Avantor. Clusters: DACH, France & Benelux, UK & Nordics.
Methods: A = supply side (restated peer revenue + others), B = demand side (R&D spend x
consumables intensity). Publish a cell only if A and B are within 15%, else show a range.
DS EMEA revenue = public-data estimate, adjustable slider (grade C).
Phase-2 placeholders: Pharma, Industry & Service, customer survey, exact DS share.

## Product categories (DS-equivalent)
Chemistry catalogue; solvents & bulk lab chemicals; analytical reagents & standards;
biology catalogue; lab consumables. Excluded: lab water, biomonitoring, testing services,
bioprocess, instruments, diagnostics.

## Share bridge
g_DS - g_M = sum_s (w_s - v_s) * g_s  [mix]  +  sum_s w_s * (g_DS_s - g_s)  [share]

## File naming in data/raw: <country>_<company>_<fiscal-year>_<document>.pdf (lowercase, hyphens, fiscal year = year covered)