"""6-step peer restatement: peer revenue -> DS-comparable EMEA Academia revenue per cluster (method A input).

    uv run python -m src.models.peer_restatement show [--online]   # offline from caches; --online fetches ECB FX once

Steps (CLAUDE.md "Peer restatement"), all plain-Python arithmetic, low/base/high carried separately:
  1 total revenue (EDGAR, A)
  2 x Europe share (disclosed geo revenue / total) x mix factor
  3 x business scope share  = (sum of in-scope sub-segment revenue x weight - intra-group double counting) / total
  4 x DS-equivalent product share;  4b: 'of which resold Merck' = result x Merck resale share (shown separately)
  5 x Academia share
  6 x cluster share, then USD -> EUR at the fiscal-year average of ECB monthly rates
Ranges are bounding: low = product of lows, high = product of highs. Output grade = worst input grade.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from src.extract.numbers import quote_contains_value
from src.facts_store import AssumptionRow, FactRow
from src.settings import PROJECT_ROOT

ASSUMPTIONS_DIR = PROJECT_ROOT / "config" / "assumptions"
FX_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "ecb"
PEERS = ["thermo_fisher", "avantor"]
GRADE_ORDER = {"A": 0, "B": 1, "C": 2}


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
    quote: str | None = None         # for A/B values: quote that must contain the base value
    unit: str = "share"

    @model_validator(mode="after")
    def _check(self):
        if self.grade not in GRADE_ORDER:
            raise ValueError(f"{self.key}: grade must be A, B or C")
        if not self.low <= self.base <= self.high:
            raise ValueError(f"{self.key}: need low <= base <= high, got {self.low}/{self.base}/{self.high}")
        if self.grade == "C" and self.low == self.high:
            raise ValueError(f"{self.key}: grade C must be a range (low < high)")
        if self.quote and not quote_contains_value(self.quote, self.base, "%" if self.unit == "%" else ""):
            raise ValueError(f"{self.key}: quote does not contain {self.base}")
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
    academia_share: Assumption

    def assumptions(self) -> list[Assumption]:
        return ([self.geo.mix_factor] + [i.weight for i in self.scope.include + self.scope.deduct]
                + [self.ds_share, self.merck_resale_share, self.academia_share])

    def evidence(self) -> list[tuple[str, str]]:
        """(key, verbatim sentence) pairs to check against the 10-K text."""
        out = [(a.key, a.evidence) for a in self.assumptions() if a.evidence]
        out += [(i.weight.key, i.evidence) for i in self.scope.include + self.scope.deduct if i.evidence]
        return out


class Market(BaseModel):
    clusters: dict[str, Assumption]
    fx: dict


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


# ---------------------------------------------------------------- restatement

Triple = tuple[float, float, float]


def _mul(a: Triple, b: Triple) -> Triple:
    return a[0] * b[0], a[1] * b[1], a[2] * b[2]


def _scale(a: Triple, k: float) -> Triple:
    return a[0] * k, a[1] * k, a[2] * k


def _worst(*grades: str) -> str:
    return max(grades, key=GRADE_ORDER.__getitem__)


@dataclass
class Step:
    n: str
    label: str
    factor: Triple | None         # share applied in this step (None for step 1)
    value: Triple                 # running value after the step (USD m, EUR m for step 6)
    grade: str
    detail: str = ""


@dataclass
class Restatement:
    peer: str
    label: str
    period: str
    steps: list[Step]
    europe_ds_academia_usd: Triple
    clusters: dict[str, Triple]           # EUR m
    clusters_merck: dict[str, Triple]     # EUR m, of which resold Merck
    fx_rate: float
    fx_source: str
    grade: str
    assumptions: list[Assumption] = field(default_factory=list)

    @property
    def total(self) -> Triple:
        vals = list(self.clusters.values())
        return tuple(sum(v[i] for v in vals) for i in range(3))  # type: ignore[return-value]

    @property
    def total_merck(self) -> Triple:
        vals = list(self.clusters_merck.values())
        return tuple(sum(v[i] for v in vals) for i in range(3))  # type: ignore[return-value]


def restate(cfg: PeerConfig, facts: list[FactRow], market: Market, fx_rate: float, fx_source: str = "") -> Restatement:
    idx = FactIndex(facts, cfg.peer)
    steps: list[Step] = []

    # 1 total revenue
    rev = idx.value(cfg.revenue)
    v: Triple = (rev, rev, rev)
    steps.append(Step("1", "Total revenue", None, v, "A", f"{cfg.period} {cfg.currency}_m"))

    # 2 Europe share
    num, den = idx.value(cfg.geo.numerator), idx.value(cfg.geo.denominator)
    ratio = num / den
    geo = _scale(cfg.geo.mix_factor.triple, ratio)
    v = _mul(v, geo)
    steps.append(Step("2", "x Europe share", geo, v, _worst("A", cfg.geo.mix_factor.grade),
                      f"{num:,.1f} / {den:,.1f} = {ratio:.1%} ({cfg.geo.numerator.period}) x mix "
                      f"{cfg.geo.mix_factor.low}/{cfg.geo.mix_factor.base}/{cfg.geo.mix_factor.high}"))

    # 3 business scope (low: include low, deduct high; high: the reverse)
    inc = [(idx.value(i.ref), i.weight) for i in cfg.scope.include]
    ded = [(idx.value(i.ref), i.weight) for i in cfg.scope.deduct]
    scope_abs = (
        sum(x * w.low for x, w in inc) - sum(x * w.high for x, w in ded),
        sum(x * w.base for x, w in inc) - sum(x * w.base for x, w in ded),
        sum(x * w.high for x, w in inc) - sum(x * w.low for x, w in ded),
    )
    scope_share = _scale(scope_abs, 1 / rev)
    v = _mul(v, scope_share)
    steps.append(Step("3", "x Business scope share", scope_share, v,
                      _worst("A", *(w.grade for _, w in inc + ded)),
                      f"in-scope {scope_abs[1]:,.1f} of {rev:,.1f} (range {scope_abs[0]:,.0f}-{scope_abs[2]:,.0f})"))

    # 4 DS-equivalent products
    v = _mul(v, cfg.ds_share.triple)
    steps.append(Step("4", "x DS-equivalent product share", cfg.ds_share.triple, v, cfg.ds_share.grade))

    # 5 Academia
    v = _mul(v, cfg.academia_share.triple)
    steps.append(Step("5", "x Academia share", cfg.academia_share.triple, v, cfg.academia_share.grade))
    europe = v

    # 6 clusters + FX
    clusters, merck = {}, {}
    for key, share in market.clusters.items():
        eur = _scale(_mul(europe, share.triple), 1 / fx_rate)
        clusters[key] = eur
        merck[key] = _mul(eur, cfg.merck_resale_share.triple)
    total = tuple(sum(c[i] for c in clusters.values()) for i in range(3))
    cluster_grade = _worst(*(s.grade for s in market.clusters.values()))
    steps.append(Step("6", "x cluster shares, USD->EUR", None, total, cluster_grade,  # type: ignore[arg-type]
                      f"sum of 3 clusters, {cfg.currency}/EUR {fx_rate:.4f} (ECB FY average)"))

    used = cfg.assumptions() + list(market.clusters.values())
    grade = _worst("A", *(a.grade for a in used))
    return Restatement(cfg.peer, cfg.label, cfg.period, steps, europe, clusters, merck, fx_rate, fx_source, grade, used)


# ---------------------------------------------------------------- outputs for the DB

@dataclass
class RunResult:
    results: list[Restatement]
    configs: list[PeerConfig]
    market: Market

    def fact_rows(self) -> list[FactRow]:
        rows = []
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for r in self.results:
            parts = {**r.clusters, "total": r.total}
            merck = {**r.clusters_merck, "total": r.total_merck}
            for region, trip in parts.items():
                for name, val, mv in zip(("low", "base", "high"), trip, merck[region]):
                    val = round(val, 1)
                    quote = (f"derived: peer_restatement {r.peer} {r.period} {region} {name} = {val:.1f} EUR_m "
                             f"(6-step chain, {r.label}; of which resold Merck {mv:.1f}; FX {r.fx_rate:.4f})")
                    rows.append(FactRow("derived", "derived:peer_restatement", now, r.peer, r.period,
                                        f"ds_academia_revenue_{name}", val, "EUR_m", region, "academia", quote, r.grade))
        return rows

    def assumption_rows(self) -> list[AssumptionRow]:
        out: dict[str, AssumptionRow] = {}
        for cfg in self.configs:
            for a in cfg.assumptions():
                k = f"{cfg.peer}.{a.key}"
                out[k] = AssumptionRow(k, a.low, a.base, a.high, a.source, a.grade, a.note)
        for a in self.market.clusters.values():
            out[f"market.{a.key}"] = AssumptionRow(f"market.{a.key}", a.low, a.base, a.high, a.source, a.grade, a.note)
        return list(out.values())


def run_all(facts: list[FactRow], online_fx: bool = False, peers: list[str] = PEERS) -> RunResult:
    market = load_market()
    configs = [load_peer(p) for p in peers]
    results = []
    for cfg in configs:
        rate, src = fx_units_per_eur(cfg.currency, int(cfg.period[2:6]), online=online_fx)
        results.append(restate(cfg, facts, market, rate, src))
    return RunResult(results, configs, market)


# ---------------------------------------------------------------- CLI

def _fmt(t: Triple, pct: bool = False) -> str:
    if pct:
        return f"{t[0]:6.1%} / {t[1]:6.1%} / {t[2]:6.1%}"
    return f"{t[0]:9,.0f} / {t[1]:9,.0f} / {t[2]:9,.0f}"


def print_result(r: Restatement) -> None:
    print(f"\n{r.label} – {r.period} – DS-comparable EMEA Academia revenue (grade {r.grade}; low / base / high)")
    for s in r.steps:
        f = _fmt(s.factor, pct=True) if s.factor else " " * 26
        unit = "EUR m" if s.n == "6" else "USD m"
        print(f"  {s.n} {s.label:<32}{f:>28}   {_fmt(s.value)} {unit}  [{s.grade}] {s.detail}")
    print("  by cluster (EUR m):")
    for k, t in r.clusters.items():
        print(f"    {k:<20}{_fmt(t)}   of which resold Merck {_fmt(r.clusters_merck[k])}")
    print(f"    {'total':<20}{_fmt(r.total)}   of which resold Merck {_fmt(r.total_merck)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--online", action="store_true", help="fetch the ECB FX average if not cached")
    args = ap.parse_args(argv)

    from src.connectors import edgar

    facts, problems = edgar.load_all(online=False)
    if problems:
        print("EDGAR cross-check problems:", problems)
    res = run_all(facts, online_fx=args.online)
    for cfg in res.configs:
        for p in check_evidence(cfg, edgar.ten_k_texts(cfg.peer)):
            print("evidence:", p)
    for r in res.results:
        print_result(r)


if __name__ == "__main__":
    main()
