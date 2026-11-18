"""Gap reconciliation: independent checks next to methods A and B, implied DS shares, A/B gap decomposition.

    uv run python -m src.models.reconciliation show

Checks (per cluster, FY2025, EUR m, Monte Carlo draws):
  merck_anchor  Merck's own Discovery market (CMD 2025 group p.14, ~EUR 80bn global 2025, grade A fact) x Europe
                share (min/median/max of TMO, Avantor, Merck Group disclosed Europe revenue shares, grade B) x cluster
                share (research base, B) x Academia weight (C) x scope overlap (C)            -> two-sided, grade C
  de_floor      German bottom-up floor from register filings: Fisher Scientific GmbH + VWR International GmbH (DACH
                cross-check anchors) + Carl Roth (FY2024 Inland x DS-category share x academic share); a lower bound
                for Germany. Scaled to DACH / the 3 clusters by Germany's share of academic research (B), so the
                scaled value is an estimate, not a hard floor                                   -> lower bound, grade C
Reconciliation (market_model.reconcile): A and B within 15% -> published as before. Otherwise the published grade-C
range is the overlap [max lower bound, min upper bound] over A, B, anchor (P10/P90) and floor (P10 only); if the
ranges do not overlap, the range is the gap between the closest bounds (every check touches it) and is flagged.
Gap decomposition: A and B with every input at its median (~ the Monte Carlo P50 case), one input moved at a time:
swing across its stated range, and the break-even value that alone would close the gap.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

import numpy as np

from src.facts_store import FactRow
from src.models.peer_restatement import (
    PCTS, Assumption, FactIndex, Market, PeerConfig, RunResult, _eur_m, _fact, evaluate, triangular,
)
from src.models.research_base import SPEND_SECTORS, Panel, fill_interior

TOTAL = "total"


def _pct(x: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.percentile(x, PCTS))  # type: ignore[return-value]


@dataclass
class Check:
    method: str                    # 'merck_anchor' | 'de_floor'
    label: str
    cluster: str
    draws: np.ndarray = field(repr=False)
    lower_only: bool = False
    grade: str = "C"
    note: str = ""

    @property
    def pct(self) -> tuple[float, float, float]:
        return _pct(self.draws)


# ---------------------------------------------------------------- Merck anchor

def europe_share(facts: list[FactRow], configs: list[PeerConfig], merck) -> tuple[Assumption, dict[str, float]]:
    """Europe share of the global market: min / median / max of disclosed Europe revenue shares (grade B)."""
    vals = {}
    for cfg in configs:
        idx = FactIndex(facts, cfg.peer)
        vals[f"{cfg.label} {cfg.geo.numerator.period}"] = idx.value(cfg.geo.numerator) / idx.value(cfg.geo.denominator)
    g = merck.disclosed["group_europe_share_q3_2026"]
    vals["Merck Group Q3 2026"] = g.base
    lo, mid, hi = min(vals.values()), statistics.median(vals.values()), max(vals.values())
    a = Assumption(key="anchor_europe_share", low=round(lo, 4), base=round(mid, 4), high=round(hi, 4), grade="B",
                   source="derived: min / median / max of disclosed Europe revenue shares – "
                          + "; ".join(f"{k} {v:.1%}" for k, v in vals.items()) + " (EDGAR XBRL, Merck release p.2)")
    return a, vals


def anchor_market_fact(facts: list[FactRow], merck) -> FactRow:
    spec = merck.anchor["market_fact"]
    hits = [r for r in facts if r.entity == spec["entity"] and r.period == spec["period"] and r.metric == spec["metric"]]
    if not hits:
        raise KeyError(f"no fact {spec} – check config/manual_facts.yaml")
    return hits[0]


def merck_anchor(facts, market: Market, merck, configs, n: int, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    fact = anchor_market_fact(facts, merck)
    size = fact.value * (1000 if fact.unit == "EUR_bn" else 1)           # EUR m
    eu, eu_vals = europe_share(facts, configs, merck)
    acad = Assumption.model_validate(merck.bridge["market_academia_weight"])
    overlap = Assumption.model_validate(merck.anchor["scope_overlap"])
    base = size * triangular(eu, n, rng) * triangular(acad, n, rng) * triangular(overlap, n, rng)
    out = {k: base * triangular(a, n, rng) for k, a in market.clusters.items()}
    out[TOTAL] = sum(out.values())
    info = {"fact": fact, "size_eur_m": size, "europe_share": eu, "europe_values": eu_vals, "academia_weight": acad,
            "scope_overlap": overlap}
    return out, info


# ---------------------------------------------------------------- German floor

def germany_floor(facts, market: Market, crosscheck, n: int, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    """Returns ({'de': floor, 'dach': scaled, 'total': scaled}, parts). Needs the DACH cross-check anchors."""
    rng = np.random.default_rng(seed)
    anc = market.dach_anchor
    by = {c.peer: c for c in crosscheck}
    if not {"thermo_fisher", "avantor"} <= set(by) or any(by[p].anchor_draws is None for p in by):
        raise KeyError("DACH cross-check anchors missing")
    period = market.reconciliation.get("carl_roth_period", "FY2024")
    roth_inland = _eur_m(_fact(facts, "carl_roth", period, "Inland"))
    roth = roth_inland * triangular(anc["carl_roth_ds_share"], n, rng) * triangular(anc["carl_roth_academic_share"], n, rng)
    parts = {"Fisher Scientific GmbH": by["thermo_fisher"].anchor_draws,
             "VWR International GmbH": by["avantor"].anchor_draws,
             f"Carl Roth ({period})": roth}
    de = sum(parts.values())
    out = {"de": de,
           "dach": de / triangular(anc["germany_share_of_dach"], n, rng),
           TOTAL: de / triangular(anc["germany_share_of_clusters"], n, rng)}
    return out, {"parts": parts, "carl_roth_inland": roth_inland, "carl_roth_period": period}


def checks(peers: RunResult, facts: list[FactRow], merck, n: int, seed: int) -> tuple[dict[str, list[Check]], dict]:
    market = peers.market
    out: dict[str, list[Check]] = {k: [] for k in list(market.clusters) + [TOTAL]}
    info: dict = {}
    try:
        anchor, info["anchor"] = merck_anchor(facts, market, merck, peers.configs, n, seed + 500)
        for k, d in anchor.items():
            out[k].append(Check("merck_anchor", "Merck anchor (CMD Discovery market)", k, d,
                                note="CMD 2025 ~EUR 80bn global Discovery market x Europe x cluster x Academia x scope"))
    except KeyError as e:
        info["anchor_error"] = str(e)
    try:
        floor, info["floor"] = germany_floor(facts, market, peers.crosscheck, n, seed + 600)
        info["floor"]["de"] = floor["de"]
        for k in ("dach", TOTAL):
            out[k].append(Check("de_floor", "German register floor (scaled)", k, floor[k], lower_only=True,
                                note="Fisher Scientific GmbH + VWR International GmbH + Carl Roth, Germany, "
                                     "scaled by Germany's share of academic research"))
    except KeyError as e:
        info["floor_error"] = str(e)
    return out, info


# ---------------------------------------------------------------- implied DS share + plausibility

@dataclass
class ShareCheck:
    method: str
    cluster: str
    share: tuple[float, float, float]
    flag: str            # '' | 'check' | 'implausible'
    reason: str


def implied_shares(ds: dict[str, np.ndarray], methods: dict[str, dict[str, np.ndarray]], peers: RunResult,
                   band: tuple[float, float]) -> list[ShareCheck]:
    """ds: cluster -> DS Academia draws; methods: name -> cluster -> market draws."""
    out = []
    for name, cells in methods.items():
        for k, m in cells.items():
            if k not in ds:
                continue
            share = _pct(ds[k] / m)
            peer = sum(r.mc_draws[k] for r in peers.results)
            resold = sum(r.mc_draws["merck_total" if k == TOTAL else f"merck_{k}"] for r in peers.results)
            room = float(np.median(m - peer + resold))
            flag, reason = "", ""
            if float(np.median(ds[k])) > room:
                flag, reason = "implausible", (f"DS alone ({np.median(ds[k]):,.0f}) exceeds the market minus Thermo "
                                               f"Fisher and Avantor plus their resold Merck ({room:,.0f}) at the P50")
            elif not band[0] <= share[1] <= band[1]:
                flag, reason = "check", f"P50 share {share[1]:.0%} outside the {band[0]:.0%}-{band[1]:.0%} band"
            out.append(ShareCheck(name, k, share, flag, reason))
    return out


# ---------------------------------------------------------------- gap decomposition (base case, scalar)

@dataclass
class GapItem:
    owner: str
    key: str
    source: str
    grade: str
    low: float
    base: float
    high: float
    gap_low: float                  # signed gap (B - A) / mean at input low
    gap_high: float
    break_even: float | None        # input value that alone closes the gap (B = A)
    in_range: bool

    @property
    def swing(self) -> float:
        return self.gap_high - self.gap_low


@dataclass
class GapResult:
    a: float
    b: float
    gap: float                      # signed (B - A) / mean, base case
    items: list[GapItem]
    prior_gap: float | None = None  # same with the prior (free) others' share


class BaseCase:
    """Scalar A and B for FY2025, inputs at their medians, with optional overrides {(owner, key): value}."""

    def __init__(self, peers: RunResult, panel: Panel, facts: list[FactRow]):
        self.peers, self.panel, self.market = peers, panel, peers.market
        self.idx = {c.peer: FactIndex(facts, c.peer) for c in peers.configs}
        self.fx = {r.peer: r.fx_rate for r in peers.results}
        d = self.market.demand
        self.year = int(self.market.supply["year"])
        from src.models.market_model import cluster_growth, demand_terms
        from src.models.taxonomy import load_taxonomy

        clusters = {k: v["countries"] for k, v in load_taxonomy()["clusters"].items()}
        self.terms = []                      # (sector, value, years_ahead, g0)
        for k, countries in clusters.items():
            g0 = cluster_growth(panel, countries, tuple(d["growth_window"]))
            self.terms += [(t.sector, t.value, t.years_ahead, g0) for t in demand_terms(panel, countries, self.year)]

    def params(self) -> list[tuple[str, Assumption]]:
        m = self.market
        out = [(c.peer, a) for c in self.peers.configs for a in c.assumptions()]
        out += [("market", a) for a in m.clusters.values()]
        d = m.demand
        out += [("market", Assumption.model_validate(x)) for x in
                [d["intensity"], d["non_rd_uplift"], d["growth_offset"], *d["sector_weights"].values(),
                 m.supply["others_share"]]]
        return [(o, a) for o, a in out if a.low < a.high]

    def ab(self, over: dict | None = None) -> tuple[float, float]:
        """Inputs at their median (tracks the Monte Carlo P50 better than the mode for skewed inputs)."""
        over = over or {}

        def v(owner):
            return lambda a, role: over.get((owner, a.key), over.get(("market", a.key), tri_median(a)))

        peers = sum(evaluate(c, self.idx[c.peer], self.market, self.fx[c.peer], v(c.peer))["total"]
                    for c in self.peers.configs)
        get = lambda x: over.get(("market", x["key"]), tri_median(Assumption.model_validate(x)))  # noqa: E731
        d, s = self.market.demand, self.market.supply
        a = peers / (1 - get(s["others_share"]))
        w = {sec: get(x) for sec, x in d["sector_weights"].items()}
        off = get(d["growth_offset"])
        spend = sum(w[sec] * val * (1 + g0 + off) ** ya for sec, val, ya, g0 in self.terms if sec in SPEND_SECTORS)
        b = get(d["intensity"]) * spend * (1 + get(d["non_rd_uplift"]))
        return a, b


def tri_median(a: Assumption) -> float:
    """Median of triangular(low, mode=base, high)."""
    lo, c, hi = a.low, a.base, a.high
    if hi <= lo:
        return c
    if c >= (lo + hi) / 2:
        return lo + ((hi - lo) * (c - lo) / 2) ** 0.5
    return hi - ((hi - lo) * (hi - c) / 2) ** 0.5


def _gap(a: float, b: float) -> float:
    return (b - a) / ((a + b) / 2)


def gap_decomposition(peers: RunResult, panel: Panel, facts: list[FactRow]) -> GapResult:
    bc = BaseCase(peers, panel, facts)
    a0, b0 = bc.ab()
    items = []
    for owner, x in bc.params():
        k = (owner, x.key)
        gl, gh = _gap(*bc.ab({k: x.low})), _gap(*bc.ab({k: x.high}))
        is_share = x.unit == "share" and x.high <= 1
        lo_s, hi_s = 0.0, (0.999 if is_share else max(10 * x.base, x.high * 2, 1e-6))
        f = lambda val: (lambda ab: ab[1] - ab[0])(bc.ab({k: val}))  # noqa: E731
        be = None
        flo, fhi = f(lo_s), f(hi_s)
        if flo * fhi < 0:
            lo, hi = lo_s, hi_s
            for _ in range(80):
                mid = (lo + hi) / 2
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            be = (lo + hi) / 2
        items.append(GapItem(owner, x.key, x.source, x.grade, x.low, x.base, x.high, gl, gh, be,
                             be is not None and x.low <= be <= x.high))
    items.sort(key=lambda i: -abs(i.swing))
    prior = None
    p = peers.market.supply.get("others_share_prior")
    if p:
        prior = _gap(*bc.ab({("market", "others_share"): tri_median(Assumption.model_validate(p))}))
    return GapResult(a0, b0, _gap(a0, b0), items, prior)


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["show"])
    ap.parse_args(argv)
    from src.models import scenario as sc

    base = sc.prepare()
    s = sc.run(base, {})
    m = s.model.market
    o, p = m.supply["others_share"], m.supply.get("others_share_prior", {})
    print(f"Others' share (method A): tenders {o['low']:.1%} / {o['base']:.1%} / {o['high']:.1%} [{o['grade']}]"
          + (f"  (prior estimate {p['low']:.0%} / {p['base']:.0%} / {p['high']:.0%})" if p else ""))
    print(f"\nMarket {s.year}, EUR m P10 / P50 / P90")
    for k in list(m.clusters) + [TOTAL]:
        a, b, r = s.cell(k, "A"), s.cell(k, "B"), s.cell(k, "reconciled")
        line = f"  {k:<20} A {a.low:6,.0f}/{a.base:6,.0f}/{a.high:6,.0f}  B {b.low:6,.0f}/{b.base:6,.0f}/{b.high:6,.0f}"
        for c in s.checks.get(k, []):
            q = c.pct
            line += f"  {c.method} {q[0]:6,.0f}/{q[1]:6,.0f}/{q[2]:6,.0f}"
        print(line)
        print(f"  {'':<20} gap {r.gap_pct:.0%} -> {r.status}: {r.low:,.0f} – {r.high:,.0f}  ({r.note})")
    fl = s.check_info.get("floor")
    if fl:
        print("\nGerman register floor (Germany, EUR m P10/P50/P90):",
              " / ".join(f"{v:,.0f}" for v in _pct(fl["de"])))
        for name, d in fl["parts"].items():
            print(f"    {name:<28} " + " / ".join(f"{v:,.0f}" for v in _pct(d)))
    an = s.check_info.get("anchor")
    if an:
        print(f"\nMerck anchor: {an['size_eur_m']:,.0f} EUR m global x Europe {an['europe_share'].low:.1%}-"
              f"{an['europe_share'].high:.1%} ({', '.join(f'{k} {v:.1%}' for k, v in an['europe_values'].items())})")
    print("\nImplied DS share of the Academia market (P10 / P50 / P90)")
    for c in s.share_checks:
        if c.cluster == TOTAL or c.flag:
            print(f"  {c.method:<14} {c.cluster:<20} {c.share[0]:5.1%} / {c.share[1]:5.1%} / {c.share[2]:5.1%}  "
                  f"{c.flag.upper():<11} {c.reason}")
    sh = s.ds.shares[TOTAL]
    print(f"  -> DS share vs reconciled market: {sh.low:.1%} – {sh.high:.1%}")
    g = s.gap
    print(f"\nA/B gap decomposition (inputs at median: A {g.a:,.0f}, B {g.b:,.0f}, signed gap {g.gap:+.0%}"
          + (f"; with the prior others' share {g.prior_gap:+.0%}" if g.prior_gap is not None else "") + ")")
    print(f"  {'input':<46}{'range':>22}{'gap at low':>12}{'at high':>9}{'break-even':>12}  in range")
    for i in g.items[:12]:
        be = "–" if i.break_even is None else f"{i.break_even:.4g}"
        print(f"  {i.owner + '.' + i.key:<46}{f'{i.low:.4g} – {i.high:.4g}':>22}{i.gap_low:>+12.0%}{i.gap_high:>+9.0%}"
              f"{be:>12}  {'YES' if i.in_range else 'no'}")


if __name__ == "__main__":
    main()
