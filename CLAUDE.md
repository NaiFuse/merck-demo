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
7. Loaders open the DB write connection only briefly: open, write a batch, close.
   Never hold a write connection during network calls or LLM calls.

## Stack
Python 3.12, uv, DuckDB (data/demo.duckdb), httpx, pydantic, pandas, Anthropic SDK
(model name from env var ANTHROPIC_MODEL), Streamlit + Plotly for the dashboard, pytest.

## Folder structure
src/connectors/   one file per source (ted, uk_fts, edgar, eurostat, oecd_msti, cordis, openalex, companies_house, inpi);
                  tender_store.py = shared tender writes (batched, rule 7); http_cache.py = polite GET + data/raw cache
src/extract/      LLM extraction from PDFs with quotes (pdf.py) + quote/number checks (numbers.py) + manual_facts.py
                  (facts read from a rendered page image when the text layer scrambles a table; fragments verified)
src/facts_store.py shared batched writes for documents/facts/assumptions (rule 7)
src/models/       taxonomy, peer restatement, research_base (country panel + cluster shares), market_model (A/B),
                  merck_ds (DS EMEA Academia estimate + share), share_bridge, scenario (slider overrides, live rerun),
                  others_share (from tenders), reconciliation (anchor, floor, implied shares, gap decomposition)
src/app/          Streamlit: main.py (nav + sidebar sliders), ui.py (formatting, rule-3 ranges), views/ (5 pages)
config/           taxonomy.yaml, research_sources.yaml (countries + API queries), assumptions/*.yaml (one per peer + market
                  + merck_ds.yaml: quoted Merck disclosures, DS estimates, bridge inputs)
data/raw/         manually downloaded PDFs (merck, peers, registers)
tests/

## Database tables
documents(id, source, url_or_path, retrieved_at)
facts(id, document_id, entity, period, metric, value, unit, region, segment, quote, grade)
assumptions(id, key, value_low, value_base, value_high, source, grade, note)
tenders(notice_id, country, buyer, cpv, value_eur, award_date, winner_name, winner_id,
        + provenance: notice_type, publication_date, buyer_legal_type, cpv_all, value, currency,
          value_field, n_winners, source_url, api_query, retrieved_at, raw_json, is_framework, title,
          source = 'ted' | 'uk_fts'; FTS notice_ids are prefixed 'FTS-')
tender_awards(notice_id, seq, winner_name, winner_id, value, currency, value_eur, alignment, grade,
              evidence, source_url, scope_note)
fx_rates(currency, month, units_per_eur, source_url, retrieved_at)
market_cells(segment, category, cluster, year, method, value_low, value_base, value_high, grade,
             + status, gap_pct, note, computed_at)  method = 'A' | 'B' | 'merck_anchor' | 'de_floor' | 'reconciled';
             status = estimate | check | published | range | range_partial; category 'all_ds' = all DS
             categories; cluster 'total' = sum of 3 clusters; values = MC P10/P50/P90 (EUR m)

## Demo scope
Segment: Academia (universities, research institutes, hospitals, government labs).
Peers: Thermo Fisher, Avantor. Clusters: DACH, France & Benelux, UK & Nordics.
Methods: A = supply side (restated peer revenue + others), B = demand side (R&D spend x
consumables intensity). Publish a cell only if A and B are within 15%, else show a range.
Market model (src/models/market_model.py, config/assumptions/market.yaml demand/supply):
  B = intensity x (HES + w_gov x GOV + w_pnp x PNP R&D spend) x (1 + non-R&D uplift), 2019-2025; spend from Eurostat
  rd_e_gerdtot, UK HES/GOV from OECD MSTI (Eurostat UK ends 2019 and predates the ONS revision), GBP at ECB annual
  average; years after a series' last observation nowcast at the cluster's 2019-23 CAGR + offset.
  A = (TMO + Avantor restated cluster revenue) / (1 - others_share), FY2025 only. others_share is derived from
  TED + UK FTS win shares (src/models/others_share.py: min/median/max of 8 views = value|count x with|without
  frameworks x distributors as others|pro rata, grade B); the market.yaml value is kept as others_share_prior.
  Checks (src/models/reconciliation.py): merck_anchor = Merck CMD 2025 Discovery market ~EUR 80bn (global, 2025,
  config/manual_facts.yaml, A) x Europe share (min/median/max of TMO, Avantor, Merck Group, B) x cluster share x
  Academia weight x scope overlap (C); de_floor = Fisher Scientific GmbH + VWR International GmbH + Carl Roth register
  filings (Germany, lower bound), scaled by Germany's research share (C).
  Reconciled: |A50 - B50| / mean <= 15% -> pooled-draw P10/P50/P90, grade B ('published'); else grade C range = overlap
  of the largest subset of A, B, anchor (P10-P90) and floor (P10, lower bound only) that overlaps (ties: more of A/B,
  then narrowest); excluded checks and their distance are stated ('range_partial').
  Gap decomposition: A and B at input medians, one input moved across its range; break-even value per input.
  Implied DS share per method; flag 'implausible' if DS > market - peers + their resold Merck, 'check' if outside
  market.yaml reconciliation.ds_share_plausible.
Cluster shares for the peer restatement and Germany's share of DACH come from src/models/research_base.py: base =
R&D-spend share of Europe (EU27 + UK, CH, NO, IS), low/high = min/max over R&D spend, researchers (headcount),
OpenAlex wet-lab publications, CORDIS HES+REC grants (H2020 2017-20 window); grade B.
DS EMEA revenue = public-data estimate, adjustable slider (grade C): DS revenue FY2025 x DS Europe share; DS Academia
= x Academia share x cluster share; DS share = DS Academia / market draws (A, B; range unless the cell is published).
Dashboard: scenario.prepare() verifies quotes once (~15 s, cached); scenario.run() reruns all Monte Carlos per slider
move (~0.05 s). Sliders move an input's base (shift for shares, scale for intensities), keep its width, stay grade C.
Tender win shares (TED + UK FTS, src/models/tender_report.py): default view = core CPV, Academia buyers only
(config/academia_buyers.yaml; hospitals count as Academia), excl. IVD winners, clinical blood-collection
consumables, analyser bundles and Merck lab water; frameworks incl. framework-like notices (>= 5 suppliers,
equal split). Groups: Thermo Fisher, Avantor, Merck, multi-brand distributors, others. Win shares, not market
shares. Negative tender values (TED -1 = not disclosed) are stored as NULL.
Phase-2 placeholders: Pharma, Industry & Service, customer survey, exact DS share.

## Product categories (DS-equivalent)
Chemistry catalogue; solvents & bulk lab chemicals; analytical reagents & standards;
biology catalogue; lab consumables. Excluded: lab water, biomonitoring, testing services,
bioprocess, instruments, diagnostics.

## Peer restatement (src/models/peer_restatement.py, config/assumptions/<peer>.yaml + market.yaml)
Each peer has its own Monte Carlo seed (market seed + peer index) so peer draws are independent.
1. Total revenue (EDGAR XBRL, A)
2. x Europe share = disclosed Europe revenue / total, x mix factor (Avantor: FY2023 geo segments, last disclosed)
3. x Business scope share = (in-scope sub-segment revenue x weight - intra-group double counting) / total
4. x DS-equivalent product share; 4b: "of which resold Merck" shown separately (channel resale is kept, not dropped)
5. x Academia share = disclosed end-market share (TMO A&G, Avantor Europe E&G) x concentration x EMEA factor
   + hospital add-on (Merck counts hospitals as Academia)
6. x cluster share (derived, research_base), USD->EUR at the FY average of ECB monthly rates
Monte Carlo (10,000 triangular draws, config/assumptions/market.yaml): P10-P90 is the default display, min/max
(bounding) shown next to it; sensitivity = one-at-a-time swing + variance share. DACH sanity check against
Fisher Scientific GmbH / VWR International GmbH register filings. Output grade = worst input grade.

## Share bridge
g_DS - g_M = sum_s (w_s - v_s) * g_s  [mix]  +  sum_s w_s * (g_DS_s - g_s)  [share]

## File naming in data/raw: <country>_<company>_<fiscal-year>_<document>.pdf (lowercase, hyphens, fiscal year = year covered)