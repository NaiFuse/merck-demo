"""Research base: country x sector x year panel of academic R&D activity, and cluster shares derived from it.

    uv run python -m src.models.research_base show [--online]   # --online fetches missing ECB GBP rates once

Inputs are facts from the four statistics connectors (Eurostat, OECD MSTI, CORDIS, OpenAlex); no network here
except the optional ECB FX fetch. All arithmetic is plain Python.

Panel rules
  R&D spend (EUR m): Eurostat rd_e_gerdtot (HES, GOV, PNP). For a (country, sector) that OECD MSTI covers (UK
    HES/GOV), the OECD series replaces Eurostat's completely (Eurostat UK stops in 2019 and predates the ONS
    2018 revision). GBP -> EUR at the ECB annual average of monthly rates.
  Researchers (headcount): Eurostat rd_p_persocc (HES, GOV), OECD for the UK.
  Publications: OpenAlex works, sum of the configured disjoint topic queries.
  EU grants: CORDIS net EC contribution to HES + REC participants, H2020 + Horizon Europe.
  Gaps: interior years linearly interpolated; for window averages, the nearest observation is carried to
  window years outside the observed span (flagged as 'carried').

Cluster shares (input to the peer restatement, step 6): each indicator gives the cluster's share of Europe
(config/research_sources.yaml europe_countries). base = R&D-spend share, low/high = min/max over all indicators.
Grade B (triangulated from 4 public sources). Same for Germany's share of DACH (DACH register anchor).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

from src.facts_store import FactRow
from src.models.taxonomy import load_taxonomy

SPEND_SECTORS = ("HES", "GOV", "PNP")
INDICATORS = ("rd_spend", "researchers", "publications", "eu_grants")
INDICATOR_LABEL = {
    "rd_spend": "R&D spend HES+GOV+PNP (Eurostat; UK: OECD MSTI)",
    "researchers": "researchers HES+GOV, headcount (Eurostat; UK: OECD MSTI)",
    "publications": "wet-lab publications, academia-type institutions (OpenAlex)",
    "eu_grants": "EU grants to HES+REC, net EC contribution (CORDIS)",
}

Series = dict[int, float]


@dataclass
class Panel:
    spend: dict[tuple[str, str], Series] = field(default_factory=dict)        # (country, sector) -> EUR m, observed
    researchers: dict[tuple[str, str], Series] = field(default_factory=dict)
    publications: dict[str, Series] = field(default_factory=dict)
    grants: dict[str, Series] = field(default_factory=dict)
    spend_source: dict[tuple[str, str], str] = field(default_factory=dict)   # 'eurostat' | 'oecd_msti'
    fx: dict[int, float] = field(default_factory=dict)                       # GBP per EUR by year
    flags: dict[tuple[str, str, int], str] = field(default_factory=dict)     # (country, sector, year) -> status flag

    def last_year(self, key: tuple[str, str]) -> int:
        return max(self.spend[key])


# ---------------------------------------------------------------- panel

def _gbp_rates(years: set[int], online: bool) -> dict[int, float]:
    from src.models.peer_restatement import fx_units_per_eur
    return {y: fx_units_per_eur("GBP", y, online=online)[0] for y in sorted(years)}


def build_panel(facts: list[FactRow], online_fx: bool = False) -> Panel:
    p = Panel()
    by_src: dict[str, dict[tuple[str, str], Series]] = defaultdict(lambda: defaultdict(dict))
    rse_src: dict[str, dict[tuple[str, str], Series]] = defaultdict(lambda: defaultdict(dict))
    gbp_years = {int(r.period) for r in facts if r.source == "oecd_msti" and r.unit.startswith("GBP")}
    p.fx = _gbp_rates(gbp_years, online_fx) if gbp_years else {}
    for r in facts:
        if r.source not in ("eurostat", "oecd_msti", "openalex", "cordis"):
            continue
        y = int(r.period)
        if r.metric == "rd_expenditure":
            v = r.value
            if r.unit.startswith("GBP"):
                if r.unit != "GBP_m":
                    raise ValueError(f"unexpected unit {r.unit}")
                v = r.value / p.fx[y]
            elif r.unit != "EUR_m":
                raise ValueError(f"unexpected unit {r.unit}")
            by_src[r.source][(r.entity, r.segment)][y] = v
            if r.extra.get("flag"):
                p.flags[(r.entity, r.segment, y)] = r.extra["flag"]
        elif r.metric == "researchers":
            rse_src[r.source][(r.entity, r.segment)][y] = r.value
        elif r.metric == "publications":
            p.publications.setdefault(r.entity, {})
            p.publications[r.entity][y] = p.publications[r.entity].get(y, 0.0) + r.value
        elif r.metric == "eu_grant_net_contribution":
            p.grants.setdefault(r.entity, {})
            p.grants[r.entity][y] = p.grants[r.entity].get(y, 0.0) + r.value
    for target, src, record in ((p.spend, by_src, True), (p.researchers, rse_src, False)):
        oecd = src.get("oecd_msti", {})
        for key, s in src.get("eurostat", {}).items():
            if key not in oecd and s:
                target[key] = dict(s)
                if record:
                    p.spend_source[key] = "eurostat"
        for key, s in oecd.items():
            target[key] = dict(s)
            if record:
                p.spend_source[key] = "oecd_msti"
    return p


def fill_interior(s: Series) -> Series:
    """Linear interpolation of missing years between the first and last observation."""
    if not s:
        return {}
    ys = sorted(s)
    out = dict(s)
    for a, b in zip(ys, ys[1:]):
        for y in range(a + 1, b):
            out[y] = s[a] + (s[b] - s[a]) * (y - a) / (b - a)
    return out


def window_mean(s: Series, y0: int, y1: int) -> tuple[float, bool]:
    """Mean over [y0, y1] of the interior-filled series; years outside the observed span take the nearest
    observation. Returns (mean, carried?). Empty series -> (0.0, False)."""
    f = fill_interior(s)
    if not f:
        return 0.0, False
    lo, hi = min(f), max(f)
    vals, carried = [], False
    for y in range(y0, y1 + 1):
        if y in f:
            vals.append(f[y])
        else:
            carried = True
            vals.append(f[hi] if y > hi else f[lo])
    return sum(vals) / len(vals), carried


def country_totals(p: Panel, indicator: str, window: tuple[int, int],
                   countries: list[str]) -> tuple[dict[str, float], set[str]]:
    """country -> window mean of the indicator (sectors summed). Also returns countries with carried values."""
    y0, y1 = window
    out, carried = {}, set()
    for c in countries:
        if indicator in ("rd_spend", "researchers"):
            src = p.spend if indicator == "rd_spend" else p.researchers
            parts = [window_mean(s, y0, y1) for (cc, _), s in src.items() if cc == c]
        else:
            src = p.publications if indicator == "publications" else p.grants
            parts = [window_mean(src.get(c, {}), y0, y1)]
        out[c] = sum(v for v, _ in parts)
        if any(cr for _, cr in parts):
            carried.add(c)
    return out, carried


@dataclass
class IndicatorShares:
    indicator: str
    window: tuple[int, int]
    shares: dict[str, float]              # group -> share of the denominator
    carried: set[str]                     # countries with values carried outside their observed span


def group_shares(p: Panel, indicators_cfg: dict, groups: dict[str, list[str]],
                 denominator: list[str]) -> list[IndicatorShares]:
    out = []
    for ind in INDICATORS:
        win = tuple(indicators_cfg[ind]["window"])
        tot, carried = country_totals(p, ind, win, denominator)
        den = sum(tot.values())
        if den <= 0:
            raise ValueError(f"{ind}: no data in window {win}")
        shares = {g: sum(tot.get(c, 0.0) for c in cs) / den for g, cs in groups.items()}
        out.append(IndicatorShares(ind, win, shares, carried & {c for cs in groups.values() for c in cs}))
    return out


# ---------------------------------------------------------------- shares as assumptions

def _assumption(key: str, group: str, ind: list[IndicatorShares], base_ind: str, what: str):
    from src.models.peer_restatement import Assumption

    vals = {s.indicator: s.shares[group] for s in ind}
    parts = "; ".join(f"{s.indicator} {s.shares[group]:.1%} ({s.window[0]}-{s.window[1]}"
                      + (f", carried: {','.join(sorted(s.carried))}" if s.carried else "") + ")" for s in ind)
    return Assumption(
        key=key, low=round(min(vals.values()), 4), base=round(vals[base_ind], 4), high=round(max(vals.values()), 4),
        grade="B", source=f"derived:research_base – {what}; base = {base_ind}, range = min/max of 4 indicators: {parts}",
        note="Indicators: " + " | ".join(f"{k}: {INDICATOR_LABEL[k]}" for k in vals))


def cluster_share_assumptions(p: Panel, sources_cfg: dict, rb_cfg: dict) -> tuple[dict, list[IndicatorShares]]:
    tax = load_taxonomy()
    groups = {k: v["countries"] for k, v in tax["clusters"].items()}
    ind = group_shares(p, rb_cfg["indicators"], groups, sources_cfg["europe_countries"])
    out = {k: _assumption(f"cluster_share_{k}", k, ind, rb_cfg["base_indicator"], f"{k} share of Europe academic R&D")
           for k in groups}
    return out, ind


def germany_share_of_dach(p: Panel, rb_cfg: dict) -> tuple:
    tax = load_taxonomy()
    dach = tax["clusters"]["dach"]["countries"]
    ind = group_shares(p, rb_cfg["indicators"], {"DE": ["DE"]}, dach)
    return _assumption("germany_share_of_dach", "DE", ind, rb_cfg["base_indicator"], "Germany share of DACH"), ind


def germany_share_of_clusters(p: Panel, rb_cfg: dict) -> tuple:
    """Germany's share of the three demo clusters (scales the German register floor to the cluster total)."""
    tax = load_taxonomy()
    allc = [c for v in tax["clusters"].values() for c in v["countries"]]
    ind = group_shares(p, rb_cfg["indicators"], {"DE": ["DE"]}, allc)
    return _assumption("germany_share_of_clusters", "DE", ind, rb_cfg["base_indicator"],
                       "Germany share of the three demo clusters"), ind


def apply_to_market(market, facts: list[FactRow], online_fx: bool = False):
    """Market with derived cluster shares (and Germany share of DACH) when market.yaml asks for it."""
    from src.connectors.eurostat import load_config

    if market.cluster_shares_from != "research_base":
        return market
    p = build_panel(facts, online_fx)
    shares, _ = cluster_share_assumptions(p, load_config(), market.research_base)
    anchor = dict(market.dach_anchor)
    if anchor:   # the DACH register cross-check scales the DACH estimate by Germany's share
        anchor["germany_share_of_dach"], _ = germany_share_of_dach(p, market.research_base)
        anchor["germany_share_of_clusters"], _ = germany_share_of_clusters(p, market.research_base)
    return market.model_copy(update={"clusters": shares, "dach_anchor": anchor})


def load_facts(online: bool = False) -> list[FactRow]:
    """All research-base facts from the connector caches (online: fetch what is missing)."""
    from src.connectors import cordis, eurostat, oecd_msti, openalex

    return (eurostat.load_all(online) + oecd_msti.load_all(online) + openalex.load_all(online)
            + cordis.load_all(online))


# ---------------------------------------------------------------- CLI

def print_shares(ind: list[IndicatorShares], groups: list[str], title: str) -> None:
    print(f"\n{title}")
    print("  " + f"{'indicator':<14}{'window':<12}" + "".join(f"{g:>20}" for g in groups) + "   carried")
    for s in ind:
        print("  " + f"{s.indicator:<14}{s.window[0]}-{s.window[1]:<7}" + "".join(f"{s.shares[g]:>20.1%}" for g in groups)
              + "   " + ",".join(sorted(s.carried)))


def main(argv: list[str] | None = None) -> None:
    from src.connectors.eurostat import load_config
    from src.models.peer_restatement import load_market

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--online", action="store_true")
    args = ap.parse_args(argv)
    market = load_market()
    p = build_panel(load_facts(), online_fx=args.online)
    shares, ind = cluster_share_assumptions(p, load_config(), market.research_base)
    print_shares(ind, list(shares), "Cluster share of Europe (EU27 + UK, CH, NO, IS) by indicator")
    for k, a in shares.items():
        print(f"  -> {a.key:<34} {a.low:.1%} / {a.base:.1%} / {a.high:.1%}  [{a.grade}]")
    de, ind_de = germany_share_of_dach(p, market.research_base)
    print_shares(ind_de, ["DE"], "Germany share of DACH by indicator")
    print(f"  -> {de.key:<34} {de.low:.1%} / {de.base:.1%} / {de.high:.1%}  [{de.grade}]")
    print("\nR&D spend series ending before 2024 (nowcast in method B):")
    for key in sorted(p.spend):
        if p.last_year(key) < 2024 and sum(p.spend[key].values()) > 0:
            print(f"  {key[0]} {key[1]} last {p.last_year(key)} ({p.spend_source[key]}) "
                  f"{p.spend[key][p.last_year(key)]:,.0f} EUR m")


if __name__ == "__main__":
    main()
