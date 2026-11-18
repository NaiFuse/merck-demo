"""Eurostat, OECD MSTI, CORDIS, OpenAlex connectors + HTTP cache: parsing, provenance quotes, secrets (offline)."""

import json
import zipfile
from datetime import datetime

import httpx

from src.connectors import cordis, eurostat, oecd_msti, openalex
from src.connectors.http_cache import PoliteClient, cached_get, read_meta
from src.facts_store import check_rows

TS = datetime(2026, 11, 18)
CFG = {
    "europe_countries": ["DE", "GB", "FR"],
    "first_year": 2022,
    "eurostat": {"base_url": "https://es/data",
                 "rd_expenditure": {"dataset": "rd_e_gerdtot", "filters": {"unit": "MIO_EUR", "sectperf": ["HES", "GOV"]}}},
    "oecd_msti": {"base_url": "https://oecd/data", "countries": {"GB": "GBR"},
                  "measures": {"H": {"metric": "rd_expenditure", "sector": "HES", "unit": "XDC"},
                               "H_RS": {"metric": "researchers", "sector": "HES", "unit": "PS"}}},
    "cordis": {"activity_types": ["HES", "REC"]},
    "openalex": {"base_url": "https://oa/works", "last_year": 2023, "institution_types": ["education", "healthcare"],
                 "queries": {"life": "primary_topic.domain.id:1|4"}},
}

# JSON-stat 2.0: dims sectperf(2) x geo(2) x time(2), row-major
JSONSTAT = {
    "id": ["freq", "sectperf", "unit", "geo", "time"], "size": [1, 2, 1, 2, 2],
    "dimension": {"freq": {"category": {"index": {"A": 0}}}, "sectperf": {"category": {"index": {"GOV": 0, "HES": 1}}},
                  "unit": {"category": {"index": {"MIO_EUR": 0}}}, "geo": {"category": {"index": {"DE": 0, "UK": 1}}},
                  "time": {"category": {"index": {"2022": 0, "2023": 1}}}},
    "value": {"0": 14697.104, "1": 15503.406, "4": 22007.356, "5": 23026.413, "7": 10239.584},
    "status": {"0": "b", "7": "p"},
}


def test_eurostat_parse_maps_codes_and_flags():
    obs = eurostat.parse_jsonstat(JSONSTAT, "rd_e_gerdtot")
    assert len(obs) == 5
    by = {(o.dim("sectperf"), o.dim("geo"), o.time): o for o in obs}
    assert by[("GOV", "DE", "2022")].value == 14697.104 and by[("GOV", "DE", "2022")].flag == "b"
    assert by[("HES", "UK", "2023")].value == 10239.584 and by[("HES", "UK", "2023")].flag == "p"
    facts = eurostat.to_facts(obs, "rd_expenditure", "https://es/data/rd_e_gerdtot?x", TS)
    assert {f.entity for f in facts} == {"DE", "GB"}                    # UK -> GB
    assert all(f.unit == "EUR_m" and f.grade == "A" for f in facts)
    check_rows(facts)                                                     # rule 1: quote contains the value
    assert "flag b" in next(f.quote for f in facts if f.period == "2022" and f.segment == "GOV")


def test_eurostat_query_params_use_eurostat_geo_codes():
    p = eurostat.query_params({"unit": "MIO_EUR", "sectperf": ["HES", "GOV"]}, ["DE", "GB", "GR"], 2015)
    assert ("geo", "UK") in p and ("geo", "EL") in p and ("sectperf", "GOV") in p and ("sinceTimePeriod", "2015") in p


def test_oecd_parse_keeps_gbp_and_status():
    text = ("REF_AREA,FREQ,MEASURE,UNIT_MEASURE,PRICE_BASE,TRANSFORMATION,TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_MULT\n"
            "GBR,A,H,XDC,V,_Z,2018,13922.18,B,6\n"
            "GBR,A,H,XDC,V,_Z,2023,17133.41,A,6\n"
            "GBR,A,H,USD_PPP,V,_Z,2023,99999,A,6\n"           # unit not configured -> skipped
            "GBR,A,H_RS,PS,_Z,_Z,2019,349015,E,0\n")
    facts = oecd_msti.parse_csv(text, CFG, "https://oecd/x", TS)
    assert [(f.period, f.unit, f.value) for f in facts] == [("2018", "GBP_m", 13922.18), ("2023", "GBP_m", 17133.41),
                                                            ("2019", "count", 349015.0)]
    assert facts[0].extra["flag"] == "B" and facts[1].extra["flag"] is None
    check_rows(facts)


def _zip(path, projects, orgs):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("project.csv", '"id";"acronym";"ecSignatureDate";"objective"\n' + "\n".join(projects))
        z.writestr("organization.csv", '"projectID";"activityType";"country";"netEcContribution"\n' + "\n".join(orgs))


def test_cordis_aggregates_by_country_year_type(tmp_path):
    zp = tmp_path / "c.zip"
    _zip(zp, ['"1";"A";"2019-05-01";"long; text\nwith newline"', '"2";"B";"2020-01-10";"x"', '"3";"C";"";"no date"'],
         ['"1";"HES";"UK";"1000,5"', '"1";"HES";"UK";"200"', '"1";"PRC";"UK";"999"', '"2";"REC";"DE";"300"',
          '"3";"HES";"DE";"50"', '"9";"HES";"DE";"7"'])
    agg = cordis.aggregate(zp, ["HES", "REC"])
    assert agg == {("GB", 2019, "HES"): {"net": 1200.5, "n": 2}, ("DE", 2020, "REC"): {"net": 300.0, "n": 1}}
    assert cordis.cached_aggregate(zp, ["HES", "REC"]) == agg
    assert cordis.cached_aggregate(zp, ["HES", "REC"]) == agg             # second call reads the cache file
    facts = cordis.to_facts(agg, "h2020", "https://cordis/x.zip", TS, {"GB"})
    assert len(facts) == 1 and facts[0].value == 1200.5 and facts[0].unit == "EUR"
    check_rows(facts)


def test_openalex_parse_and_params():
    d = {"meta": {"groups_count": 3}, "group_by": [{"key": "https://openalex.org/countries/GB", "count": 10},
                                                   {"key": "https://openalex.org/countries/US", "count": 99},
                                                   {"key": "unknown", "count": 5}]}
    assert openalex.parse_groups(d) == {"GB": 10, "US": 99}
    p = openalex.params_for(CFG, 2023, "life")
    assert p["filter"] == ("publication_year:2023,authorships.institutions.type:education|healthcare,"
                           "primary_topic.domain.id:1|4")
    facts = openalex.to_facts({"GB": 10, "US": 99}, 2023, "life", "https://oa/w", TS, {"GB", "DE"})
    assert [f.entity for f in facts] == ["GB"]
    check_rows(facts)


def test_cached_get_never_stores_secret_params(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"ok": 1})

    client = PoliteClient(min_interval=0, client=httpx.Client(transport=httpx.MockTransport(handler)))
    body, meta = cached_get(client, "https://api.x/works", {"filter": "a:1"}, tmp_path, ".json",
                            secret_params={"api_key": "SECRET-KEY", "mailto": "me@example.org"})
    assert json.loads(body) == {"ok": 1}
    assert "SECRET-KEY" in seen[0]                                          # sent ...
    assert "SECRET" not in meta["url"] and "example.org" not in meta["url"]  # ... never stored
    stored = "".join(p.read_text() for p in tmp_path.iterdir())
    assert "SECRET" not in stored and "example.org" not in stored
    body2, meta2 = cached_get(None, "https://api.x/works", {"filter": "a:1"}, tmp_path, ".json")   # offline hit
    assert body2 == body and len(seen) == 1


def test_polite_client_retries_on_429():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429 if len(calls) < 3 else 200, text="ok")

    c = PoliteClient(min_interval=0, sleep=lambda s: None, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.get("https://x").text == "ok" and len(calls) == 3


def test_cordis_meta_for_hand_downloaded_zip(tmp_path):
    zp = tmp_path / "c.zip"
    _zip(zp, [], [])
    m = cordis.meta_for(zp, "https://cordis/c.zip")
    assert m["url"] == "https://cordis/c.zip" and read_meta(zp)["retrieved_at"] == m["retrieved_at"]
