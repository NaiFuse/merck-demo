"""Market model: EMEA Academia DS-type market per cluster and year – method A (supply), B (demand), reconciled.

    uv run python -m src.models.market_model show [--online]   # offline from caches; --online fetches ECB FX once

Method B (demand side), per cluster and year (config/assumptions/market.yaml 'demand'):
    B = intensity x sum_countries (HES + w_gov x GOV + w_pnp x PNP) x (1 + non-R&D uplift)
    R&D spend from the research base (Eurostat; UK via OECD MSTI). Years after a series' last observation are
    nowcast with the cluster's observed nominal CAGR over growth_window (+ growth_offset).
Method A (supply side), FY2025 only (the peer restatement year):
    A = (Thermo Fisher + Avantor restated cluster revenue) / (1 - others_share)
Reconciliation per cluster and year where both exist: publish if |A50 - B50| / mean(A50, B50) <= 15% (taxonomy
publish_tolerance) -> value from the pooled A+B draws, grade B (two independent methods agree); else a range
min(P10) .. max(P90) with no point value, grade C. B-only years are written as method B rows only.

Monte Carlo: 10,000 independent triangular draws per input (seeded); peers' draws come from the restatement.
Cells store P10 / P50 / P90 as value_low / value_base / value_high. All arithmetic is numpy / plain Python.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.facts_store import AssumptionRow, FactRow
from src.models.peer_restatement import PCTS, Assumption, Market, RunResult, _grade, triangular
from src.models.research_base import SPEND_SECTORS, Panel, fill_interior
from src.models.taxonomy import load_taxonomy

SEGMENT = "academia"
CATEGORY = "all_ds"          # all DS-equivalent categories combined (neither method splits by category)
TOTAL = "total"              # sum of the three demo clusters


def _pct(x: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.percentile(x, PCTS))  # type: ignore[return-value]


def _a(d: dict) -> Assumption:
    return Assumption.model_validate(d)


# ---------------------------------------------------------------- method B

@dataclass
class DemandTerm:
    country: str
    sector: str
    value: float          # EUR m: observed/interpolated value, or the last observation for nowcast years
    years_ahead: int      # 0 = observed or interpolated; k > 0 = nowcast k years past the last observation


def demand_terms(p: Panel, countries: list[str], year: int) -> list[DemandTerm]:
    out = []
    for (c, s), series in p.spend.items():
        if c not in countries or s not in SPEND_SECTORS:
            continue
        f = fill_interior(series)
        last = max(f)
        if year in f:
            out.append(DemandTerm(c, s, f[year], 0))
        elif year > last:
            out.append(DemandTerm(c, s, f[last], year - last))
        # year before the series starts: no observation to extend backwards, left out
    return out


def cluster_growth(p: Panel, countries: list[str], window: tuple[int, int]) -> float:
    """Nominal CAGR of the cluster's HES+GOV+PNP spend over the window, on series observed at both ends."""
    y0, y1 = window
    a = b = 0.0
    for (c, s), series in p.spend.items():
        f = fill_interior(series)
        if c in countries and y0 in f and y1 in f:
            a += f[y0]
            b += f[y1]
    if a <= 0:
        raise ValueError(f"no spend observed in {window} for {countries}")
    return (b / a) ** (1 / (y1 - y0)) - 1


@dataclass
class DemandCell:
    cluster: str
    year: int
    spend_weighted_base: float     # EUR m, sector-weighted spend at base weights / base growth
    spend_raw: float               # EUR m, unweighted HES+GOV+PNP (base growth for nowcast years)
    nowcast_share: float           # share of spend_raw that is nowcast
    growth_base: float
    draws: np.ndarray = field(repr=False)
    grade: str = "C"               # worst input grade (spend facts are A)

    @property
    def pct(self) -> tuple[float, float, float]:
        return _pct(self.draws)


def method_b(p: Panel, market: Market, n: int, seed: int) -> list[DemandCell]:
    d = market.demand
    rng = np.random.default_rng(seed)
    intensity, uplift, offset = (_a(d[k]) for k in ("intensity", "non_rd_uplift", "growth_offset"))
    weights = {s: _a(w) for s, w in d["sector_weights"].items()}
    draw = {"intensity": triangular(intensity, n, rng), "uplift": triangular(uplift, n, rng),
            **{f"w_{s}": triangular(w, n, rng) for s, w in weights.items()}}
    clusters = {k: v["countries"] for k, v in load_taxonomy()["clusters"].items()}
    base_grade = _grade("A", intensity.grade, uplift.grade, *(w.grade for w in weights.values()))
    y0, y1 = d["years"]
    out = []
    for k, countries in clusters.items():
        g0 = cluster_growth(p, countries, tuple(d["growth_window"]))
        g = g0 + triangular(offset, n, rng)
        for year in range(y0, y1 + 1):
            terms = demand_terms(p, countries, year)
            weighted = np.zeros(n)
            raw = nowcast = w_base = 0.0
            for t in terms:
                v = t.value * (1 + g) ** t.years_ahead
                weighted += draw[f"w_{t.sector}"] * v
                vb = t.value * (1 + g0) ** t.years_ahead
                raw += vb
                w_base += weights[t.sector].base * vb
                nowcast += vb if t.years_ahead else 0.0
            cell = weighted * draw["intensity"] * (1 + draw["uplift"])
            grade = _grade(base_grade, offset.grade) if nowcast else base_grade
            out.append(DemandCell(k, year, w_base, raw, nowcast / raw if raw else 0.0, g0, cell, grade))
    for year in range(y0, y1 + 1):
        cs = [c for c in out if c.year == year]
        out.append(DemandCell(TOTAL, year, sum(c.spend_weighted_base for c in cs), sum(c.spend_raw for c in cs),
                              sum(c.nowcast_share * c.spend_raw for c in cs) / sum(c.spend_raw for c in cs),
                              float("nan"), sum(c.draws for c in cs), _grade(*(c.grade for c in cs))))
    return out


# ---------------------------------------------------------------- method A

@dataclass
class SupplyCell:
    cluster: str
    year: int
    peers: dict[str, tuple[float, float, float]]     # peer -> P10/P50/P90 EUR m
    draws: np.ndarray = field(repr=False)
    grade: str = "C"                                  # worst of the peer restatements and others_share

    @property
    def pct(self) -> tuple[float, float, float]:
        return _pct(self.draws)


def method_a(peers: RunResult, market: Market, n: int, seed: int) -> list[SupplyCell]:
    s = market.supply
    others = _a(s["others_share"])
    year = int(s["year"])
    for r in peers.results:
        if int(r.period[2:6]) != year:
            raise ValueError(f"{r.peer}: restatement period {r.period} != supply year {year}")
        if len(next(iter(r.mc_draws.values()))) != n:
            raise ValueError(f"{r.peer}: {len(next(iter(r.mc_draws.values())))} draws, need {n}")
    rng = np.random.default_rng(seed)
    o = triangular(others, n, rng)
    grade = _grade(others.grade, *(r.grade for r in peers.results))
    out = []
    for k in list(peers.market.clusters) + [TOTAL]:
        peer_sum = sum(r.mc_draws[k] for r in peers.results)
        out.append(SupplyCell(k, year, {r.peer: r.mc[k] for r in peers.results}, peer_sum / (1 - o), grade))
    return out


# ---------------------------------------------------------------- reconciliation

@dataclass
class MarketCell:
    cluster: str
    year: int
    method: str                    # 'A' | 'B' | 'reconciled'
    low: float | None
    base: float | None
    high: float | None
    grade: str
    status: str                    # 'estimate' | 'check' | 'published' | 'range' | 'range_partial'
    gap_pct: float | None = None   # |A50 - B50| / mean(A50, B50), reconciled rows only
    note: str = ""
    members: list = field(default_factory=list)     # reconciled range: methods/checks whose ranges overlap
    excluded: dict = field(default_factory=dict)    # method -> distance to the range, EUR m (+ above, - below)


def reconcile(a: SupplyCell, b: DemandCell, tolerance: float, checks: list | None = None) -> MarketCell:
    """Published cells are grade B (two independent methods agree), or A if both inputs were A already.
    Otherwise grade C: the overlap of A, B and the independent checks (reconciliation.Check: P10/P90, lower-only
    checks contribute their P10); if they do not overlap, the gap between the closest bounds (flagged)."""
    a50, b50 = a.pct[1], b.pct[1]
    gap = abs(a50 - b50) / ((a50 + b50) / 2)
    if gap <= tolerance:
        lo, mid, hi = _pct(np.concatenate([a.draws, b.draws]))
        grade = "A" if a.grade == b.grade == "A" else "B"
        return MarketCell(a.cluster, a.year, "reconciled", lo, mid, hi, grade, "published", gap,
                          f"A and B agree within {tolerance:.0%} (gap {gap:.1%}); pooled A+B draws P10/P50/P90")
    head = f"A {a50:,.0f} vs B {b50:,.0f} EUR m: gap {gap:.0%} > {tolerance:.0%}"
    if not checks:
        lo, hi = min(a.pct[0], b.pct[0]), max(a.pct[2], b.pct[2])
        return MarketCell(a.cluster, a.year, "reconciled", lo, None, hi, "C", "range", gap,
                          f"{head} – range min P10 .. max P90")
    iv = {"A": (a.pct[0], a.pct[2]), "B": (b.pct[0], b.pct[2])}
    for c in checks:
        q = c.pct
        iv[c.method] = (q[0], float("inf") if c.lower_only else q[2])
    sub, lo, hi = largest_overlap(iv)
    excluded = {m: (iv[m][0] - hi if iv[m][0] > hi else iv[m][1] - lo) for m in iv if m not in sub}
    parts = [f"{head} – overlap of {', '.join(sub)}"]
    for m, d in excluded.items():
        parts.append(f"{m} excluded: its P10 {iv[m][0]:,.0f} is {d:,.0f} above the overlap" if d > 0
                     else f"{m} excluded: its P90 {iv[m][1]:,.0f} is {-d:,.0f} below the overlap")
    return MarketCell(a.cluster, a.year, "reconciled", lo, None, hi, "C", "range_partial" if excluded else "range",
                      gap, "; ".join(parts), list(sub), excluded)


def largest_overlap(iv: dict[str, tuple[float, float]]) -> tuple[list[str], float, float]:
    """Largest subset of intervals with a common overlap; ties -> more of the primary methods A, B, then narrowest.
    Returns (members, low, high). A and B always overlap with themselves, so a subset of size >= 1 exists."""
    from itertools import combinations

    names = list(iv)
    best = None
    for r in range(len(names), 0, -1):
        for sub in combinations(names, r):
            lo = max(iv[m][0] for m in sub)
            hi = min(iv[m][1] for m in sub)
            if lo < hi and hi != float("inf"):
                key = (r, len({"A", "B"} & set(sub)), -(hi - lo))   # primary methods before checks
                if best is None or key > best[0]:
                    best = (key, list(sub), lo, hi)
        if best is not None:
            break
    return best[1], best[2], best[3]


@dataclass
class ModelResult:
    demand: list[DemandCell]
    supply: list[SupplyCell]
    cells: list[MarketCell]
    market: Market
    tolerance: float
    checks: dict = field(default_factory=dict)      # cluster -> [reconciliation.Check]

    def cell_rows(self) -> list[dict]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return [{"segment": SEGMENT, "category": CATEGORY, "cluster": c.cluster, "year": c.year, "method": c.method,
                 "value_low": _r(c.low), "value_base": _r(c.base), "value_high": _r(c.high), "grade": c.grade,
                 "status": c.status, "gap_pct": None if c.gap_pct is None else round(c.gap_pct, 4), "note": c.note,
                 "computed_at": now} for c in self.cells]

    def assumption_rows(self) -> list[AssumptionRow]:
        d, s = self.market.demand, self.market.supply
        items = [d["intensity"], d["non_rd_uplift"], d["growth_offset"], *d["sector_weights"].values(), s["others_share"]]
        out = []
        for x in items:
            a = _a(x)
            out.append(AssumptionRow(f"market_model.{a.key}", a.low, a.base, a.high, a.source, a.grade, a.note))
        if s.get("others_share_prior"):
            a = _a(s["others_share_prior"])
            out.append(AssumptionRow("market_model.others_share_prior", a.low, a.base, a.high,
                                     "prior estimate, replaced by tender win shares – " + a.source, a.grade, a.note))
        for c in self.demand:
            if c.cluster != TOTAL and c.year == d["years"][1]:
                out.append(AssumptionRow(
                    f"market_model.nowcast_growth_{c.cluster}", None, round(c.growth_base, 4), None,
                    f"derived:research_base – nominal CAGR of HES+GOV+PNP spend {d['growth_window'][0]}-"
                    f"{d['growth_window'][1]} (Eurostat; UK OECD MSTI)", "A"))
        return out

    def fact_rows(self) -> list[FactRow]:
        """Derived spend base per cluster/year (EUR m) so the dashboard can show B's driver."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = []
        for c in self.demand:
            v = round(c.spend_raw, 1)
            rows.append(FactRow("derived", "derived:market_model", now, "research_base", str(c.year),
                                "academic_rd_spend", v, "EUR_m", c.cluster, SEGMENT,
                                f"derived: market_model HES+GOV+PNP R&D spend {c.cluster} {c.year} = {v:.1f} EUR_m "
                                f"(nowcast share {c.nowcast_share:.0%})", "A" if c.nowcast_share == 0 else "C"))
        return rows


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def run(peers: RunResult, panel: Panel, checks: dict | None = None) -> ModelResult:
    market = peers.market
    mc = market.monte_carlo
    n, seed = int(mc["draws"]), int(mc["seed"])
    tol = float(load_taxonomy()["publish_tolerance"])
    demand = method_b(panel, market, n, seed + 100)
    supply = method_a(peers, market, n, seed + 200)
    cells: list[MarketCell] = []
    for b in demand:
        cells.append(MarketCell(b.cluster, b.year, "B", *b.pct, b.grade, "estimate",
                                note=f"spend {b.spend_raw:,.0f} EUR m (nowcast {b.nowcast_share:.0%})"))
    by_b = {(b.cluster, b.year): b for b in demand}
    for a in supply:
        cells.append(MarketCell(a.cluster, a.year, "A", *a.pct, a.grade, "estimate",
                                note="peers P50: " + ", ".join(f"{k} {v[1]:,.0f}" for k, v in a.peers.items())))
        for c in (checks or {}).get(a.cluster, []):
            q = c.pct
            cells.append(MarketCell(a.cluster, a.year, c.method, q[0], q[1], q[2], c.grade, "check", note=c.note))
        if (a.cluster, a.year) in by_b:
            cells.append(reconcile(a, by_b[(a.cluster, a.year)], tol, (checks or {}).get(a.cluster)))
    return ModelResult(demand, supply, cells, market, tol, checks or {})


def build(facts: list[FactRow], online_fx: bool = False) -> tuple[RunResult, ModelResult]:
    """facts = EDGAR + PDF + research-base facts. Returns (peer restatement, market model)."""
    from src.models import peer_restatement, research_base

    from src.models import merck_ds, reconciliation

    peers = peer_restatement.run_all(facts, online_fx=online_fx)
    panel = research_base.build_panel(facts, online_fx)
    mc = peers.market.monte_carlo
    checks, _ = reconciliation.checks(peers, facts, merck_ds.load_config(), int(mc["draws"]), int(mc["seed"]))
    return peers, run(peers, panel, checks)


# ---------------------------------------------------------------- CLI

def _f(t) -> str:
    return " / ".join("    –" if v is None else f"{v:6,.0f}" for v in t)


def print_result(peers: RunResult, res: ModelResult) -> None:
    year = int(res.market.supply["year"])
    print("\nCluster shares used by the peer restatement (derived, research base)")
    for k, a in res.market.clusters.items():
        print(f"  {k:<20} {a.low:.1%} / {a.base:.1%} / {a.high:.1%}  [{a.grade}]")
    print("\nMethod B – demand side, EUR m, P10 / P50 / P90  (spend = HES+GOV+PNP R&D, EUR m)")
    for c in res.demand:
        if c.year >= year - 2:
            print(f"  {c.cluster:<20} {c.year}  {_f(c.pct)}   spend {c.spend_raw:9,.0f}  nowcast {c.nowcast_share:4.0%}"
                  + (f"  CAGR {c.growth_base:.1%}" if c.cluster != TOTAL else ""))
    print(f"\nMethod A – supply side FY{year}, EUR m, P10 / P50 / P90")
    for a in res.supply:
        print(f"  {a.cluster:<20} {_f(a.pct)}   peers P50 " + ", ".join(f"{k} {v[1]:,.0f}" for k, v in a.peers.items()))
    print(f"\nReconciliation {year} (publish if |A-B| / mean <= {res.tolerance:.0%} on P50s)")
    b_by = {(c.cluster, c.year): c for c in res.demand}
    for c in res.cells:
        if c.method != "reconciled":
            continue
        b = b_by[(c.cluster, c.year)]
        a = next(x for x in res.supply if x.cluster == c.cluster)
        implied = a.pct[1] / (b.spend_weighted_base * (1 + _a(res.market.demand["non_rd_uplift"]).base))
        print(f"  {c.cluster:<20} A {a.pct[1]:6,.0f}  B {b.pct[1]:6,.0f}  gap {c.gap_pct:5.0%}  -> {c.status:<9} "
              f"[{c.grade}] {_f((c.low, c.base, c.high))}   intensity implied by A {implied:.2%} "
              f"(assumed {_a(res.market.demand['intensity']).base:.1%})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--online", action="store_true", help="fetch missing ECB FX averages")
    args = ap.parse_args(argv)
    from src.facts_store import base_facts

    peers, res = build(base_facts(), online_fx=args.online)
    print_result(peers, res)


if __name__ == "__main__":
    main()
