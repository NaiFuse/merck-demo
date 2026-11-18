"""6-step peer restatement: peer revenue -> DS-comparable EMEA Academia revenue per cluster (method A input).

    uv run python -m src.models.peer_restatement show [--online]   # offline from caches; --online fetches ECB FX once

Steps (CLAUDE.md "Peer restatement"), all plain-Python / numpy arithmetic:
  1 total revenue (EDGAR, A)
  2 x Europe share (disclosed geo revenue / total) x mix factor
  3 x business scope share  = (sum of in-scope sub-segment revenue x weight - intra-group double counting) / total
  4 x DS-equivalent product share;  4b: 'of which resold Merck' = result x Merck resale share (shown separately)
  5 x Academia share = disclosed end-market share x concentration x EMEA factor + hospital add-on
      (or a single estimated share, for configs without disclosed end-market data)
  6 x cluster share, then USD -> EUR at the fiscal-year average of ECB monthly rates

Ranges: min/max are bounding (product of lows / highs); the default display is the P10-P90 of a Monte Carlo over
all inputs (independent triangular(low, base, high) draws). Output grade = worst input grade.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from pydantic import BaseModel, model_validator

from src.extract.numbers import quote_contains_value, unit_multiplier
from src.facts_store import AssumptionRow, FactRow
from src.settings import PROJECT_ROOT

ASSUMPTIONS_DIR = PROJECT_ROOT / "config" / "assumptions"
FX_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "ecb"
PEERS = ["thermo_fisher", "avantor"]
GRADE_ORDER = {"A": 0, "B": 1, "C": 2}
PCTS = (10, 50, 90)


# ---------------------------------------------------------------- config models

class Assumption(BaseModel):
    key: str
    low: float
    base: float
    high: float
    grade: str
    source: str
    note: str | None = None
    evidence: str | None = None      # verbatim sentence from the peer's 10-K (checked against cached text)
    quote: str | None = None         # verbatim quote from source_file/page that contains quote_value (or base)
    quote_value: float | None = None # number as printed in the quote (e.g. 14 for '14%')
    source_file: str | None = None   # data/raw path of the PDF the quote is from
    page: int | None = None
    unit: str = "share"

    @model_validator(mode="after")
    def _check(self):
        if self.grade not in GRADE_ORDER:
            raise ValueError(f"{self.key}: grade must be A, B or C")
        if not self.low <= self.base <= self.high:
            raise ValueError(f"{self.key}: need low <= base <= high, got {self.low}/{self.base}/{self.high}")
        if self.grade == "C" and self.low == self.high:
            raise ValueError(f"{self.key}: grade C must be a range (low < high)")
        if self.quote:
            v = self.quote_value if self.quote_value is not None else self.base
            if not quote_contains_value(self.quote, v, "%" if (self.unit == "%" or self.quote_value is not None) else ""):
                raise ValueError(f"{self.key}: quote does not contain {v}")
        return self

    @property
    def triple(self) -> tuple[float, float, float]:
        return self.low, self.base, self.high


class FactRef(BaseModel):
    period: str
    region: str | None = None
    segment: str | None = None
    abs: bool = False


class ScopeItem(BaseModel):
    ref: FactRef
    weight: Assumption
    evidence: str | None = None


class Geo(BaseModel):
    numerator: FactRef
    denominator: FactRef
    mix_factor: Assumption


class Scope(BaseModel):
    include: list[ScopeItem]
    deduct: list[ScopeItem] = []


class Academia(BaseModel):
    """academia share = eg_share x concentration x emea_factor + hospital_addon"""
    eg_share: Assumption
    concentration: Assumption
    emea_factor: Assumption
    hospital_addon: Assumption

    def parts(self) -> list[Assumption]:
        return [self.eg_share, self.concentration, self.emea_factor, self.hospital_addon]


class PeerConfig(BaseModel):
    peer: str
    label: str
    period: str
    currency: str
    tenk_url: str
    revenue: FactRef
    geo: Geo
    scope: Scope
    ds_share: Assumption
    merck_resale_share: Assumption
    academia_share: Assumption | None = None   # single estimate (legacy / no disclosure)
    academia: Academia | None = None           # composite from disclosed end-market data

    @model_validator(mode="after")
    def _one_academia(self):
        if (self.academia_share is None) == (self.academia is None):
            raise ValueError(f"{self.peer}: give exactly one of academia_share / academia")
        return self

    def academia_parts(self) -> list[Assumption]:
        return self.academia.parts() if self.academia else [self.academia_share]

    def assumptions(self) -> list[Assumption]:
        return ([self.geo.mix_factor] + [i.weight for i in self.scope.include + self.scope.deduct]
                + [self.ds_share, self.merck_resale_share] + self.academia_parts())

    def evidence(self) -> list[tuple[str, str]]:
        """(key, verbatim sentence) pairs to check against the 10-K text."""
        out = [(a.key, a.evidence) for a in self.assumptions() if a.evidence]
        out += [(i.weight.key, i.evidence) for i in self.scope.include + self.scope.deduct if i.evidence]
        return out


class Market(BaseModel):
    clusters: dict[str, Assumption] = {}       # filled at run time when cluster_shares_from = 'research_base'
    fx: dict
    monte_carlo: dict = {"draws": 10000, "seed": 20261118}
    dach_anchor: dict[str, Assumption] = {}
    cluster_shares_from: str | None = None     # 'research_base' -> src/models/research_base.apply_to_market
    research_base: dict = {}
    demand: dict = {}                          # method B inputs (src/models/market_model.py)
    supply: dict = {}                          # method A inputs


def load_peer(peer: str, directory: Path = ASSUMPTIONS_DIR) -> PeerConfig:
    with open(directory / f"{peer}.yaml", encoding="utf-8") as f:
        return PeerConfig.model_validate(yaml.safe_load(f))


def load_market(directory: Path = ASSUMPTIONS_DIR) -> Market:
    with open(directory / "market.yaml", encoding="utf-8") as f:
        return Market.model_validate(yaml.safe_load(f))


def check_evidence(cfg: PeerConfig, tenk_texts: dict[str, str]) -> list[str]:
    """Problems: evidence sentences not found in the cached 10-K. Empty = all found. Skipped if 10-K not cached."""
    from src.extract.pdf import normalise

    text = tenk_texts.get(cfg.tenk_url)
    if text is None:
        return [f"{cfg.peer}: 10-K not cached ({cfg.tenk_url}); evidence unchecked"]
    norm = normalise(text)
    return [f"{cfg.peer}.{k}: evidence not found in 10-K: {e[:80]}" for k, e in cfg.evidence() if normalise(e) not in norm]


def check_source_quotes(assumptions: list[Assumption], root: Path = PROJECT_ROOT) -> list[str]:
    """Rule 1 for disclosed inputs: each quote must be on its stated page of its source PDF."""
    from src.extract.pdf import find_quote_pages, page_texts

    problems, cache = [], {}
    for a in assumptions:
        if not (a.quote and a.source_file):
            continue
        path = root / a.source_file
        if not path.exists():
            problems.append(f"{a.key}: source file missing {a.source_file}")
            continue
        pages = cache.setdefault(path, page_texts(path))
        found = find_quote_pages(a.quote, pages)
        if a.page not in found:
            problems.append(f"{a.key}: quote not on page {a.page} of {a.source_file} (found on {found})")
    return problems


# ---------------------------------------------------------------- facts

class FactIndex:
    def __init__(self, rows: list[FactRow], entity: str, metric: str = "revenue"):
        self._by = {(r.period, r.region, r.segment): r for r in rows if r.entity == entity and r.metric == metric}
        self.entity = entity

    def get(self, ref: FactRef) -> FactRow:
        key = (ref.period, ref.region, ref.segment)
        if key not in self._by:
            raise KeyError(f"{self.entity}: no revenue fact for {key} – run `python -m src.connectors.edgar fetch`")
        return self._by[key]

    def value(self, ref: FactRef) -> float:
        v = self.get(ref).value
        return abs(v) if ref.abs else v


# ---------------------------------------------------------------- FX

def fx_units_per_eur(currency: str, year: int, online: bool = False, cache_dir: Path = FX_CACHE_DIR) -> tuple[float, str]:
    """Average of the 12 ECB monthly rates for the year. Cached in data/raw/ecb/. Returns (rate, source_url)."""
    path = cache_dir / f"{currency}_{year}.json"
    if path.exists():
        d = json.loads(path.read_text())
    elif online:
        from src.connectors import ecb_fx
        rates = [r for r in ecb_fx.fetch_monthly_rates({currency}, f"{year}-01") if r.month.startswith(str(year))]
        if len(rates) != 12:
            raise RuntimeError(f"ECB returned {len(rates)} monthly {currency} rates for {year}, need 12")
        d = {"currency": currency, "year": year, "source_url": rates[0].source_url,
             "retrieved_at": rates[0].retrieved_at.isoformat(), "monthly": {r.month: r.units_per_eur for r in rates}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))
    else:
        raise FileNotFoundError(f"{path} missing – run `python -m src.models.peer_restatement show --online` once")
    monthly = list(d["monthly"].values())
    return sum(monthly) / len(monthly), d["source_url"]


# ---------------------------------------------------------------- core evaluation (scalars or numpy arrays)

Triple = tuple[float, float, float]
ValueFn = Callable[[Assumption, str], "float | np.ndarray"]   # (assumption, role) -> value; role 'f' | 'inc' | 'ded'


def evaluate(cfg: PeerConfig, idx: FactIndex, market: Market, fx_rate: float, val: ValueFn) -> dict:
    """One pass through the 6 steps. val() supplies each input: base, bound, or an array of draws."""
    rev = idx.value(cfg.revenue)
    ratio = idx.value(cfg.geo.numerator) / idx.value(cfg.geo.denominator)
    geo = ratio * val(cfg.geo.mix_factor, "f")
    inc = sum(idx.value(i.ref) * val(i.weight, "inc") for i in cfg.scope.include)
    ded = sum(idx.value(i.ref) * val(i.weight, "ded") for i in cfg.scope.deduct)
    scope = (inc - ded) / rev
    ds = val(cfg.ds_share, "f")
    if cfg.academia:
        a = cfg.academia
        acad = (val(a.eg_share, "f") * val(a.concentration, "f") * val(a.emea_factor, "f")
                + val(a.hospital_addon, "f"))
    else:
        acad = val(cfg.academia_share, "f")
    europe = rev * geo * scope * ds * acad
    clusters = {k: europe * val(s, "f") / fx_rate for k, s in market.clusters.items()}
    total = sum(clusters.values())
    merck = val(cfg.merck_resale_share, "f")
    return {"rev": rev, "ratio": ratio, "geo": geo, "scope": scope, "scope_abs": inc - ded, "ds": ds, "acad": acad,
            "europe": europe, "clusters": clusters, "total": total, "merck": merck}


def _base(a: Assumption, role: str) -> float:
    return a.base


def _low(a: Assumption, role: str) -> float:      # lowest output: deductions at their high
    return a.high if role == "ded" else a.low


def _high(a: Assumption, role: str) -> float:
    return a.low if role == "ded" else a.high


def triangular(a: Assumption, n: int, rng: np.random.Generator) -> np.ndarray:
    if a.low == a.high:
        return np.full(n, a.base)
    return rng.triangular(a.low, a.base, a.high, n)


def _uid(a: Assumption, owner: str) -> str:
    return f"{owner}.{a.key}"


# ---------------------------------------------------------------- results

def _grade(*grades: str) -> str:
    return max(grades, key=GRADE_ORDER.__getitem__)


@dataclass
class Step:
    n: str
    label: str
    factor: Triple | None         # share applied in this step (min / base / max)
    value: Triple                 # running value after the step (USD m, EUR m for step 6)
    grade: str
    detail: str = ""


@dataclass
class Sensitivity:
    key: str
    label: str
    swing: float              # total EUR m at input high minus at input low (others at base)
    var_share: float          # squared correlation of the input's draws with the total (first-order share)


@dataclass
class Restatement:
    peer: str
    label: str
    period: str
    steps: list[Step]
    europe_ds_academia_usd: Triple
    clusters: dict[str, Triple]           # EUR m min / base / max
    clusters_merck: dict[str, Triple]     # EUR m, of which resold Merck
    fx_rate: float
    fx_source: str
    grade: str
    assumptions: list[Assumption] = field(default_factory=list)
    mc: dict[str, tuple[float, float, float]] = field(default_factory=dict)   # name -> (P10, P50, P90)
    mc_draws: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    sensitivity: list[Sensitivity] = field(default_factory=list)

    @property
    def total(self) -> Triple:
        vals = list(self.clusters.values())
        return tuple(sum(v[i] for v in vals) for i in range(3))  # type: ignore[return-value]

    @property
    def total_merck(self) -> Triple:
        vals = list(self.clusters_merck.values())
        return tuple(sum(v[i] for v in vals) for i in range(3))  # type: ignore[return-value]


def restate(cfg: PeerConfig, facts: list[FactRow], market: Market, fx_rate: float, fx_source: str = "",
            n_draws: int | None = None, seed: int | None = None) -> Restatement:
    idx = FactIndex(facts, cfg.peer)
    lo, b, hi = (evaluate(cfg, idx, market, fx_rate, f) for f in (_low, _base, _high))

    def t(key):
        return lo[key], b[key], hi[key]

    run = lambda k: (lo[k], b[k], hi[k])  # noqa: E731
    rev = b["rev"]
    a_parts = cfg.academia_parts()
    acad_detail = (f"{cfg.academia.eg_share.base:.0%} disclosed ({cfg.academia.eg_share.key}) x concentration x "
                   f"EMEA factor + hospital add-on" if cfg.academia else "")
    v1 = (rev, rev, rev)
    v2 = tuple(rev * g for g in t("geo"))
    v3 = tuple(x * s for x, s in zip(v2, t("scope")))
    v4 = tuple(x * s for x, s in zip(v3, t("ds")))
    v5 = run("europe")
    steps = [
        Step("1", "Total revenue", None, v1, "A", f"{cfg.period} {cfg.currency}_m"),
        Step("2", "x Europe share", t("geo"), v2, _grade("A", cfg.geo.mix_factor.grade),
             f"{idx.value(cfg.geo.numerator):,.1f} / {idx.value(cfg.geo.denominator):,.1f} = {b['ratio']:.1%} "
             f"({cfg.geo.numerator.period}) x mix {cfg.geo.mix_factor.low}/{cfg.geo.mix_factor.base}/{cfg.geo.mix_factor.high}"),
        Step("3", "x Business scope share", t("scope"), v3,
             _grade("A", *(i.weight.grade for i in cfg.scope.include + cfg.scope.deduct)),
             f"in-scope {b['scope_abs']:,.1f} of {rev:,.1f} (range {lo['scope_abs']:,.0f}-{hi['scope_abs']:,.0f})"),
        Step("4", "x DS-equivalent product share", t("ds"), v4, cfg.ds_share.grade),
        Step("5", "x Academia share", t("acad"), v5, _grade(*(a.grade for a in a_parts)), acad_detail),
        Step("6", "x cluster shares, USD->EUR", None, run("total"),
             _grade(*(s.grade for s in market.clusters.values())),
             f"sum of 3 clusters, {cfg.currency}/EUR {fx_rate:.4f} (ECB FY average)"),
    ]
    clusters = {k: (lo["clusters"][k], b["clusters"][k], hi["clusters"][k]) for k in market.clusters}
    merck = {k: tuple(c * m for c, m in zip(v, (cfg.merck_resale_share.low, cfg.merck_resale_share.base,
                                                 cfg.merck_resale_share.high))) for k, v in clusters.items()}
    used = cfg.assumptions() + list(market.clusters.values())
    res = Restatement(cfg.peer, cfg.label, cfg.period, steps, v5, clusters, merck, fx_rate, fx_source,
                      _grade("A", *(a.grade for a in used)), used)
    if n_draws:
        simulate(res, cfg, idx, market, fx_rate, n_draws, seed)
    return res


# ---------------------------------------------------------------- Monte Carlo + sensitivity

def _param_owners(cfg: PeerConfig, market: Market) -> list[tuple[str, Assumption]]:
    return [(cfg.peer, a) for a in cfg.assumptions()] + [("market", a) for a in market.clusters.values()]


def simulate(res: Restatement, cfg: PeerConfig, idx: FactIndex, market: Market, fx_rate: float,
             n: int = 10000, seed: int | None = None) -> None:
    rng = np.random.default_rng(seed)
    params = _param_owners(cfg, market)
    draws = {_uid(a, o): triangular(a, n, rng) for o, a in params}
    owner = {id(a): o for o, a in params}
    out = evaluate(cfg, idx, market, fx_rate, lambda a, role: draws[_uid(a, owner[id(a)])])
    merck_total = out["total"] * out["merck"]
    series = {**out["clusters"], "total": out["total"], "merck_total": merck_total,
              **{f"merck_{k}": v * out["merck"] for k, v in out["clusters"].items()}}
    res.mc = {k: tuple(float(x) for x in np.percentile(v, PCTS)) for k, v in series.items()}
    res.mc_draws = {k: np.asarray(v) for k, v in series.items()}

    # one-at-a-time swing (others at base) + first-order variance share from the draws
    sens = []
    total = out["total"]
    for o, a in params:
        if a.low == a.high:
            continue
        def at(x, target=a):
            return evaluate(cfg, idx, market, fx_rate,
                            lambda q, role: x if q is target else q.base)["total"]
        swing = at(a.high) - at(a.low)
        r = np.corrcoef(draws[_uid(a, o)], total)[0, 1]
        sens.append(Sensitivity(_uid(a, o), a.source[:70], float(swing), float(r * r)))
    res.sensitivity = sorted(sens, key=lambda s: -abs(s.swing))


# ---------------------------------------------------------------- DACH cross-check (sanity anchor)

@dataclass
class AnchorCheck:
    peer: str
    anchor_entity: str
    estimate_dach: tuple[float, float, float]      # EUR m P10/P50/P90
    estimate_de: tuple[float, float, float]        # Germany-equivalent of the estimate
    anchor: tuple[float, float, float]             # register-based Germany anchor P10/P50/P90
    inputs: str
    scope_differences: list[str]
    rule: str
    plausible: bool
    verdict: str


def _fact(rows: list[FactRow], entity: str, period: str, segment: str | None, metric: str = "revenue") -> FactRow:
    hits = [r for r in rows if r.entity == entity and r.period == period and r.metric == metric
            and (r.segment or r.region) == segment]
    if not hits:
        raise KeyError(f"no fact {entity} {period} {metric} {segment} – run `python -m src.extract.pdf run`")
    return min(hits, key=lambda r: unit_multiplier(r.unit))   # most precise


def _eur_m(r: FactRow) -> float:
    return r.value * unit_multiplier(r.unit) / 1e6


def dach_crosscheck(results: list[Restatement], facts: list[FactRow], market: Market,
                    period: str = "FY2025", seed: int | None = None) -> list[AnchorCheck]:
    anc = market.dach_anchor
    if not anc:
        return []
    by_peer = {r.peer: r for r in results}
    n = len(next(iter(by_peer.values())).mc_draws.get("dach", [])) or 10000
    rng = np.random.default_rng(seed)
    d = {k: triangular(a, n, rng) for k, a in anc.items()}
    out = []

    def pct(x):
        return tuple(float(v) for v in np.percentile(x, PCTS))

    if "thermo_fisher" in by_peer:
        e = "fisher_scientific_gmbh"
        cons, chem = _eur_m(_fact(facts, e, period, "Labor Verbrauchsgüter")), _eur_m(_fact(facts, e, period, "Chemikalien"))
        total, inland = _eur_m(_fact(facts, e, period, None)), _eur_m(_fact(facts, e, period, "Inland"))
        anchor = (cons + chem) * (inland / total) * d["fisher_academic_share"]
        est = by_peer["thermo_fisher"].mc_draws["dach"]
        est_de = est * d["germany_share_of_dach"]
        ok = np.percentile(est_de, 90) >= np.percentile(anchor, 10)
        ratio = float(np.median(est_de) / np.median(anchor))
        out.append(AnchorCheck(
            "thermo_fisher", "Fisher Scientific GmbH", pct(est), pct(est_de), pct(anchor),
            f"(Labor Verbrauchsgüter {cons:,.1f} + Chemikalien {chem:,.1f}) x Inland {inland:,.1f}/{total:,.1f} "
            f"x academic share {anc['fisher_academic_share'].low:.0%}-{anc['fisher_academic_share'].high:.0%}",
            ["Fisher Scientific GmbH is TMO's German distribution company only: TMO's direct entities (Life Technologies,"
             " Thermo Electron etc.) are not in it, so the anchor is a LOWER bound for TMO's German DS-type academia sales",
             "register categories are broader than DS categories (lab consumables include e.g. safety items) – pushes "
             "the anchor up", "Germany only; the estimate covers DE+AT+CH (scaled by Germany's share of DACH)",
             "the register 'Academic' segment may exclude hospitals, which Merck counts as Academia"],
            "lower bound: estimate (DE-equivalent) P90 must be >= anchor P10", bool(ok),
            (f"plausible: estimate {ratio:.1f}x the channel-only lower bound, i.e. it implies TMO's direct businesses "
             f"sell ~{ratio - 1:.1f}x its German channel to DS-type academia" + (" – CHECK, high" if ratio > 3 else ""))
            if ok else "IMPLAUSIBLE: estimate below TMO's German channel alone"))
    if "avantor" in by_peer:
        e = "vwr_international_gmbh"
        chem, labtech = _eur_m(_fact(facts, e, period, "Chemikalien")), _eur_m(_fact(facts, e, period, "Labortechnik und Sonstiges"))
        total, de = _eur_m(_fact(facts, e, period, None)), _eur_m(_fact(facts, e, period, "Deutschland"))
        anchor = (chem + labtech * d["vwr_lab_consumables_share"]) * (de / total) * d["vwr_academic_share"]
        est = by_peer["avantor"].mc_draws["dach"]
        est_de = est * d["germany_share_of_dach"]
        lo_e, hi_e = np.percentile(est_de, [10, 90])
        lo_a, hi_a = np.percentile(anchor, [10, 90])
        ok = lo_e <= hi_a and lo_a <= hi_e
        out.append(AnchorCheck(
            "avantor", "VWR International GmbH", pct(est), pct(est_de), pct(anchor),
            f"(Chemikalien {chem:,.1f} + Labortechnik {labtech:,.1f} x consumables "
            f"{anc['vwr_lab_consumables_share'].low:.0%}-{anc['vwr_lab_consumables_share'].high:.0%}) x Deutschland "
            f"{de:,.1f}/{total:,.1f} x academia {anc['vwr_academic_share'].low:.0%}-{anc['vwr_academic_share'].high:.0%}",
            ["VWR International GmbH is Avantor's main German lab company (channel + own chemicals), so it is a "
             "like-for-like anchor; Avantor's Bioscience Production sites are not in scope of either",
             "'Labortechnik und Sonstiges' mixes equipment, consumables and other – consumables share is assumed",
             "Germany only (VWR GmbH also books ~33% foreign sales, removed via Deutschland/total)",
             "academia share is not disclosed for VWR GmbH; anchor uses Avantor's Europe end-market mix"],
            "two-sided: estimate (DE-equivalent) P10-P90 must overlap the anchor P10-P90", bool(ok),
            "plausible (ranges overlap)" if ok else "IMPLAUSIBLE: estimate and anchor ranges do not overlap"))
    return out


# ---------------------------------------------------------------- outputs for the DB

@dataclass
class RunResult:
    results: list[Restatement]
    configs: list[PeerConfig]
    market: Market
    crosscheck: list[AnchorCheck] = field(default_factory=list)

    def fact_rows(self) -> list[FactRow]:
        rows = []
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for r in self.results:
            parts = {**r.clusters, "total": r.total}
            merck = {**r.clusters_merck, "total": r.total_merck}
            for region, trip in parts.items():
                vals = list(zip(("low", "base", "high"), trip, merck[region]))
                if r.mc:
                    vals += list(zip(("p10", "p50", "p90"), r.mc[region], [None] * 3))
                for name, val, mv in vals:
                    val = round(val, 1)
                    extra = f"; of which resold Merck {mv:.1f}" if mv is not None else "; Monte Carlo percentile"
                    quote = (f"derived: peer_restatement {r.peer} {r.period} {region} {name} = {val:.1f} EUR_m "
                             f"(6-step chain, {r.label}{extra}; FX {r.fx_rate:.4f})")
                    rows.append(FactRow("derived", "derived:peer_restatement", now, r.peer, r.period,
                                        f"ds_academia_revenue_{name}", val, "EUR_m", region, "academia", quote, r.grade))
        return rows

    def assumption_rows(self) -> list[AssumptionRow]:
        out: dict[str, AssumptionRow] = {}
        for cfg in self.configs:
            for a in cfg.assumptions():
                k = f"{cfg.peer}.{a.key}"
                src = a.source + (f" [{a.source_file} p.{a.page}: \"{a.quote}\"]" if a.source_file else "")
                out[k] = AssumptionRow(k, a.low, a.base, a.high, src, a.grade, a.note)
        for a in list(self.market.clusters.values()) + list(self.market.dach_anchor.values()):
            out[f"market.{a.key}"] = AssumptionRow(f"market.{a.key}", a.low, a.base, a.high, a.source, a.grade, a.note)
        return list(out.values())


def run_all(facts: list[FactRow], online_fx: bool = False, peers: list[str] = PEERS) -> RunResult:
    """facts must include the research-base facts when market.yaml derives cluster shares from them."""
    from src.models import research_base

    market = research_base.apply_to_market(load_market(), facts, online_fx)
    if not market.clusters:
        raise ValueError("market.yaml has no cluster shares (set cluster_shares_from: research_base or list clusters)")
    configs = [load_peer(p) for p in peers]
    problems = check_source_quotes([a for c in configs for a in c.assumptions()])
    if problems:
        raise ValueError("assumption quotes failed verification: " + "; ".join(problems))
    mc = market.monte_carlo
    results = []
    for i, cfg in enumerate(configs):
        rate, src = fx_units_per_eur(cfg.currency, int(cfg.period[2:6]), online=online_fx)
        seed = None if mc.get("seed") is None else mc["seed"] + i   # own stream per peer: independent draws
        results.append(restate(cfg, facts, market, rate, src, n_draws=mc.get("draws"), seed=seed))
    try:
        cross = dach_crosscheck(results, facts, market, seed=mc.get("seed"))
    except KeyError:
        cross = []   # register facts not extracted yet
    return RunResult(results, configs, market, cross)


# ---------------------------------------------------------------- CLI

def _fmt(t, pct: bool = False) -> str:
    if pct:
        return f"{t[0]:6.1%} / {t[1]:6.1%} / {t[2]:6.1%}"
    return f"{t[0]:7,.0f} / {t[1]:7,.0f} / {t[2]:7,.0f}"


def print_result(r: Restatement) -> None:
    print(f"\n{r.label} – {r.period} – DS-comparable EMEA Academia revenue (grade {r.grade})")
    print("  waterfall: min / base / max")
    for s in r.steps:
        f = _fmt(s.factor, pct=True) if s.factor else " " * 26
        unit = "EUR m" if s.n == "6" else "USD m"
        print(f"  {s.n} {s.label:<32}{f:>28}   {_fmt(s.value)} {unit}  [{s.grade}] {s.detail}")
    print("  EUR m          P10 – P90 (default)     P50     |  min / base / max          of which resold Merck (P50)")
    for k in list(r.clusters) + ["total"]:
        mn = r.total if k == "total" else r.clusters[k]
        p = r.mc.get(k, (float("nan"),) * 3)
        merck_p50 = r.mc["merck_total" if k == "total" else f"merck_{k}"][1]
        print(f"    {k:<20} {p[0]:6,.0f} – {p[2]:<6,.0f}        {p[1]:6,.0f}     |  {_fmt(mn)}     {merck_p50:6,.0f}")
    print(f"  P10–P90 width {r.mc['total'][2] / max(r.mc['total'][0], 1e-9):.1f}x vs min–max {r.total[2] / max(r.total[0], 1e-9):.1f}x")


def print_sensitivity(results: list[Restatement], top: int = 3) -> None:
    print("\nSensitivity – inputs that drive the total range (swing = total at input high minus at low, others at base; "
          "variance share = first-order share of the Monte Carlo variance)")
    for r in results:
        print(f"  {r.label}:")
        for s in r.sensitivity[:top]:
            print(f"    {s.key:<46} swing {s.swing:+7,.0f} EUR m   variance share {s.var_share:5.0%}   {s.label}")
        rest = sum(s.var_share for s in r.sensitivity[top:])
        print(f"    {'(all other inputs)':<46} {'':>19}   variance share {rest:5.0%}")


def print_crosscheck(checks: list[AnchorCheck]) -> None:
    if not checks:
        print("\nDACH cross-check: register facts missing – run `python -m src.extract.pdf run`")
        return
    print("\nDACH cross-check against the peers' German register filings (FY2025, EUR m, P10 / P50 / P90)")
    for c in checks:
        print(f"  {c.peer} vs {c.anchor_entity}: {'OK  ' if c.plausible else 'FLAG'} {c.verdict}")
        print(f"    estimate DACH            {_fmt(c.estimate_dach)}")
        print(f"    estimate Germany-equiv.  {_fmt(c.estimate_de)}")
        print(f"    anchor (Germany)         {_fmt(c.anchor)}   = {c.inputs}")
        print(f"    rule: {c.rule}")
        for s in c.scope_differences:
            print(f"    scope: {s}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--online", action="store_true", help="fetch the ECB FX average if not cached")
    args = ap.parse_args(argv)

    from src.connectors import edgar
    from src.extract import pdf
    from src.models import research_base

    facts, problems = edgar.load_all(online=False)
    if problems:
        print("EDGAR cross-check problems:", problems)
    facts += pdf.verify_all_cached()[0] + research_base.load_facts()
    res = run_all(facts, online_fx=args.online)
    for cfg in res.configs:
        for p in check_evidence(cfg, edgar.ten_k_texts(cfg.peer)):
            print("evidence:", p)
    for r in res.results:
        print_result(r)
    print_sensitivity(res.results)
    print_crosscheck(res.crosscheck)


if __name__ == "__main__":
    main()
